from typing import List, Optional, Callable, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import nvtx

from dualpipe import DualPipe, set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.utils import WeightGradStore, run_backward
from packaging.version import Version as PkgVersion

class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = F.linear(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)

        def grad_weight_fn():
            weight.grad += grad_output.flatten(0, -2).T @ input.flatten(0, -2)

        if WeightGradStore.enabled:
            WeightGradStore.put(grad_weight_fn)
        else:
            grad_weight_fn()
        grad_input = grad_output @ weight
        return grad_input, None


class MyLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LinearFunc.apply(input, self.weight)

class _AllToAll(torch.autograd.Function):
  @staticmethod
  def forward(ctx, group, input, output_split_sizes=None, input_split_sizes=None, skip_comm=False):
      ctx.group = group
      ctx.output_split_sizes = output_split_sizes
      ctx.input_split_sizes = input_split_sizes

      world_size = torch.distributed.get_world_size(group=group)
      # Bypass the function if we are using only 1 GPU.
      if world_size == 1:
          return input

      input = input.contiguous()
      if output_split_sizes is None:
          # Equal split (all2all)
          output = torch.empty_like(input)
      else:
          # Unequal split (all2all-v)
          output = input.new_empty(
              size=[sum(output_split_sizes)] + list(input.size()[1:]),
              dtype=input.dtype,
              device=torch.cuda.current_device(),
          )
      if not skip_comm:
          torch.distributed.all_to_all_single(
              output,
              input,
              output_split_sizes=output_split_sizes,
              input_split_sizes=input_split_sizes,
              group=group,
          )
      return output

  @staticmethod
  def backward(ctx, *grad_output):
      return (
          None,
          _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
          None,
          None,
          None,
      )

class PipelineStage(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = MyLinear(hidden_size, hidden_size * 4, bias=False)
        self.linear2 = MyLinear(hidden_size * 4, hidden_size, bias=False)
        self.linear3 = MyLinear(hidden_size, hidden_size, bias=False)

    def _yield_and_detach(self, x):
        yield x
        new_x = x.detach().clone().requires_grad_()
        assert x._base is None
        x.data = torch.Tensor()
        yield new_x
        return new_x

    @nvtx.annotate("PipelineStage.forward", color="blue")
    def _forward(self, x: torch.Tensor, skip_comm=False) -> torch.Tensor:
        global alltoall_group
        x = self.linear1(x)
        x = F.gelu(x)
        x = yield from self._yield_and_detach(x)

        x = _AllToAll.apply(alltoall_group, x, None, None, skip_comm)
        x = yield from self._yield_and_detach(x)

        x = self.linear2(x)
        x = yield from self._yield_and_detach(x)

        x = _AllToAll.apply(alltoall_group, x, None, None, skip_comm)
        x = yield from self._yield_and_detach(x)

        x = self.linear3(x)
        return x

    #  def forward(self, x, outputs_list):
    #      x = self.linear1(x)
    #      x = F.gelu(x)
    #      x = self.linear2(x)
    #      return x

    def forward(self, x, outputs_list):
        forward_iter = self._forward(x)
        while True:
            try:
                result = next(forward_iter)
                outputs_list.append([result])
            except StopIteration as e:
                x = e.value
                break
        return x

    @classmethod
    @nvtx.annotate("overlaped_forward_backward", color="purple")
    def overlaped_forward_backward(
        cls,
        module0: "PipelineStage",
        inputs0: List[torch.Tensor],
        criterion0: Optional[Callable],
        labels0: Optional[List[torch.Tensor]],
        outputs_list0: List[torch.Tensor],
        module1: "PipelineStage",
        loss1: Optional[torch.Tensor],
        outputs1: Optional[List[torch.Tensor]],
        output_grads1: Optional[List[torch.Tensor]],
        outputs_list1: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        You should implement custom forward-backward overlap strategy.
        The code below is just an example.
        """
        global alltoall_group
        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        else:
            run_backward(outputs1, output_grads1)

        forward_iter = module0._forward(*inputs0, True)

        # backward alltoall2 and forward stage1
        output_grads = [t.grad for t in outputs_list1.pop()]
        outputs = outputs_list1.pop()
        outputs_list1[-1][0].grad = torch.empty_like(outputs_list1[-1][0])
        input_grads = outputs_list1[-1][0].grad
        handle = torch.distributed.all_to_all_single(
            input_grads,
            output_grads[0],
            group=alltoall_group,
            async_op=True,
        )
        for i in range(2):
            result = next(forward_iter)
            outputs_list0.append([result])
        handle.wait()

        # forward alltoall1 and backward stage2
        for i in range(2):
            result = next(forward_iter)
            outputs_list0.append([result])
        handle = torch.distributed.all_to_all_single(
            outputs_list0[-1][0],
            outputs_list0[-3][0],
            group=alltoall_group,
            async_op=True,
        )
        output_grads = [t.grad for t in outputs_list1.pop()]
        outputs = outputs_list1.pop()
        run_backward(tuple(outputs), tuple(output_grads))
        handle.wait()

        # backward alltoall1 and forward stage2
        output_grads = [t.grad for t in outputs_list1.pop()]
        outputs = outputs_list1.pop()
        outputs_list1[-1][0].grad = torch.empty_like(outputs_list1[-1][0])
        input_grads = outputs_list1[-1][0].grad
        handle = torch.distributed.all_to_all_single(
            input_grads,
            output_grads[0],
            group=alltoall_group,
            async_op=True,
        )
        for i in range(2):
            result = next(forward_iter)
            outputs_list0.append([result])
        handle.wait()

        # forward alltoall2 and backward stage1
        for i in range(2):
            result = next(forward_iter)
            outputs_list0.append([result])
        handle = torch.distributed.all_to_all_single(
            outputs_list0[-1][0],
            outputs_list0[-3][0],
            group=alltoall_group,
            async_op=True,
        )
        output_grads = [t.grad for t in outputs_list1.pop()]
        outputs = outputs_list1.pop()
        run_backward(tuple(outputs), tuple(output_grads))
        handle.wait()

        # forward stage3
        try:
            next(forward_iter)
        except StopIteration as e:
            outputs0 = e.value

        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        if criterion0 is not None:
            loss0 = criterion0(*outputs0, *labels0)
        else:
            loss0 = None
        return outputs0, loss0


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target).clone()


def ref_step(x, l, model, chunks):
    ys, losses = [], []
    for micro_x, micro_l in zip(x.chunk(chunks), l.chunk(chunks)):
        micro_y = model(micro_x)
        loss = criterion(micro_y, micro_l)
        loss.backward()
        ys.append(micro_y)
        losses.append(loss)
    y = torch.cat(ys, 0)
    loss = torch.stack(losses)
    return loss, y


def cal_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff

########################################################################################
try:
    _torch_version = PkgVersion(torch.__version__)
except:
    # This is a WAR for building docs, where torch is not actually imported
    _torch_version = PkgVersion("0.0.0")
_te_version = None


def get_torch_version():
    """Get pytorch version from __version__; if not available use pip's. Use caching."""

    def get_torch_version_str():
        import torch

        if hasattr(torch, '__version__'):
            return str(torch.__version__)
        else:
            return version("torch")

    global _torch_version
    if _torch_version is None:
        _torch_version = PkgVersion(get_torch_version_str())
    return _torch_version

def get_torch_version():
    """Get torch version from __version__."""

    global _torch_version
    return _torch_version


def is_torch_min_version(version, check_equality=True):
    """Check if minimum version of `torch` is installed."""
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)

def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'use_local_synchronization': use_local_synchronization,
        'group_desc': group_desc,
    }
    if not is_torch_min_version('2.4.0'):
        kwargs.pop('group_desc')
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop('timeout')
    return torch.distributed.new_group(**kwargs)
########################################################################################

def main(rank, pp_size):
    is_first_rank = rank == 0
    is_last_rank = rank == pp_size - 1
    dist.init_process_group(backend='nccl', init_method="env://", world_size=pp_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    torch.manual_seed(233)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    global alltoall_group
    alltoall_group = create_group([rank, pp_size - 1 - rank], group_desc="alltoall group")

    num_chunks = 20
    micro_batch_size = 3
    seq_len = 256
    hidden_size = 512
    if is_first_rank:
        print(f"{pp_size=}, {num_chunks=}, {seq_len=}, {hidden_size=}", flush=True)
    set_p2p_tensor_shapes([(micro_batch_size, seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.float32)

    # Create a model and partition it for each process
    full_modules = nn.Sequential(*[PipelineStage(hidden_size) for _ in range(pp_size)])

    # Full inputs
    full_x = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)
    full_l = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)

    # Reference step
    #loss_ref, output_ref = ref_step(full_x, full_l, full_modules, num_chunks)

    # DualPipe
    local_full_modules = nn.Sequential(full_modules[rank], full_modules[pp_size - 1 - rank])
    local_modules = nn.Sequential(PipelineStage(hidden_size), PipelineStage(hidden_size))
    local_modules[0].load_state_dict(local_full_modules[0].state_dict())
    local_modules[1].load_state_dict(local_full_modules[1].state_dict())
    dualpipe_model = DualPipe(local_modules)

    # DualPipe inputs
    if is_first_rank:
        x = full_x.chunk(2)[0]
        l = full_l.chunk(2)[1]
    elif is_last_rank:
        x = full_x.chunk(2)[1]
        l = full_l.chunk(2)[0]
    else:
        x = None
        l = None

    # Training step
    loss, outputs = dualpipe_model.step(x, num_chunks=num_chunks, criterion=criterion, labels=(l,), return_outputs=False)

    ## Check loss
    #if is_first_rank:
    #    assert torch.equal(loss, loss_ref.chunk(2)[1])
    #elif is_last_rank:
    #    assert torch.equal(loss, loss_ref.chunk(2)[0])
    #else:
    #    assert loss is None
    #assert outputs is None

    ## Check grads
    #for (p0, p1) in zip(local_modules[0].parameters(), local_modules[1].parameters()):
    #    p0all = torch.empty(pp_size, *p0.shape)
    #    p1all = torch.empty(pp_size, *p1.shape)
    #    dist.all_gather_into_tensor(p0all, p0.grad)
    #    dist.all_gather_into_tensor(p1all, p1.grad)
    #    p0.grad += p1all[pp_size - 1 - rank]
    #    p1.grad += p0all[pp_size - 1 - rank]
    #for ((n, p), p_ref) in zip(local_modules.named_parameters(), local_full_modules.parameters()):
    #    assert cal_diff(p.grad, p_ref.grad) < 1e-13
    #dualpipe_model.zero_grad()

    ## Inference step
    #with torch.no_grad():
    #    loss, outputs = dualpipe_model.step(x, num_chunks=num_chunks, criterion=criterion, labels=(l,), return_outputs=True)

    ## Check loss and outputs
    #if is_first_rank:
    #    assert torch.equal(loss, loss_ref.chunk(2)[1])
    #    assert torch.equal(outputs, output_ref.chunk(2)[1])
    #elif is_last_rank:
    #    assert torch.equal(loss, loss_ref.chunk(2)[0])
    #    assert torch.equal(outputs, output_ref.chunk(2)[0])
    #else:
    #    assert loss is None
    #    assert outputs is None


def test_dualpipe(ngpus):
    torch.multiprocessing.spawn(main, args=(ngpus, ), nprocs=ngpus, daemon=False)


if __name__ == "__main__":
    #num_gpus = torch.cuda.device_count() // 2 * 2
    #for ngpus in range(num_gpus, 0, -2):
    #    test_dualpipe(ngpus)
    test_dualpipe(4)
