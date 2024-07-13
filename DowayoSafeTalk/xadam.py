#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

""" Optimizer
"""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch import distributed as dist
import pdb
from lr_schedulers import SCHEDULES
# from ..utils import get_logger


def adamw(data,
          out_data,
          next_m,
          next_v,
          grad,
          lr,
          beta1,
          beta2,
          eps,
          grad_scale,  # combined_scale, g = g/scale
          step,
          eps_mode=1,  # self.eps_mode, esp inside sqrt:0, outside: 1, only update with momentum: 2
          bias_correction=0,
          weight_decay=0):
    if bias_correction > 0:
        lr *= bias_correction
    beta1_ = 1 - beta1
    beta2_ = 1 - beta2
    grad = grad.float()
    if grad_scale != 1:
        grad *= 1 / grad_scale
    next_m.mul_(beta1).add_(beta1_, grad)
    # admax
    admax = eps_mode >> 4
    eps_mode = eps_mode & 0xF
    if admax > 0:
        torch.max(next_v.mul_(beta2), grad.abs().to(next_v), out=next_v)
        update = next_m / (next_v + eps)
    else:
        next_v.mul_(beta2).addcmul_(beta2_, grad, grad)
        if eps_mode == 0:
            update = (next_m) * (next_v + eps).rsqrt()
        elif eps_mode == 1:
            update = (next_m) / (next_v.sqrt() + eps)
        else:  # =2
            update = next_m.clone()
    if weight_decay > 0:
        update.add_(weight_decay, data)

    data.add_(-lr, update)
    if (out_data is not None) and len(out_data) > 0:
        out_data.copy_(data)


class XAdam(Optimizer):
    """Implements optimized version of Adam algorithm with weight decay fix.
    Params:
      lr: learning rate
      warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
      t_total: total number of training steps for the learning
        rate schedule, -1  means constant learning rate. Default: -1
      schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
      b1: Adams b1. Default: 0.9
      b2: Adams b2. Default: 0.999
      e: Adams epsilon. Default: 1e-6
      weight_decay_rate: Weight decay. Default: 0.01
      max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
      with_radam: Whether to enable radam. Default: False
      radam_th: RAdam threshold for tractable variance. Default: 4
      opt_type: The type of optimizer, [adam, admax], default: adam
    """

    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-8, weight_decay_rate=0.01,
                 lr_ends=0,
                 max_grad_norm=1.0,
                 with_radam=False,
                 radam_th=4,
                 opt_type=None,
                 rank=-1):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        self.defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                             b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                             lr_ends=lr_ends,
                             max_grad_norm=max_grad_norm,
                             with_radam=with_radam, radam_th=radam_th)
        self.opt_type = opt_type.lower() if opt_type is not None else ""
        self.rank = rank
        super().__init__(params, self.defaults)

    def step(self, grad_scale=1, lr_scale=1):
        """Performs a single optimization step.

        Arguments:
          grad_scale: divid grad by grad_scale
          lr_scale: scale learning rate by bs_scale
        """
        if 'global_step' not in self.state:
            self.state['global_step'] = 0
        for group in self.param_groups:
            lr_sch = self.get_group_lr_sch(group, self.state['global_step'])
            if group['rank'] == self.rank or group['rank'] < 0 or self.rank < 0:
                for param in group['params']:
                    self.update_param(group, param, grad_scale, lr_scale)

        self.state['global_step'] += 1
        self.last_grad_scale = grad_scale
        handels = []
        for group in self.param_groups:
            if group['rank'] >= 0 and self.rank >= 0:
                # sync
                for param in group['params']:
                    out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
                    if out_p is not None:
                        h = torch.distributed.broadcast(out_p, group['rank'], async_op=True)
                    else:
                        h = torch.distributed.broadcast(param.data, group['rank'], async_op=True)
                    handels.append(h)

        for h in handels:
            if h is not None:
                h.wait()

        return lr_sch

    def get_group_lr_sch(self, group, steps):
        if group['t_total'] > 0:
            schedule_fct = SCHEDULES[group['schedule']]
            lr_scheduled = schedule_fct(steps, group['t_total'], group['warmup'], group['lr_ends'])
        else:
            lr_scheduled = 1
        return lr_scheduled

    def update_param(self, group, param, grad_scale, lr_scale):
        grad = param.grad
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        state = self.get_state(param)
        lr_sch = self.get_group_lr_sch(group, state['step'])
        lr = group['lr'] * lr_scale * lr_sch
        next_m, next_v = state['next_m'], state['next_v']
        beta1, beta2 = group['b1'], group['b2']
        state['step'] += 1

        # Support for RAdam
        t = (state['step'] - 1) + 1
        eps_mode = 1
        if group['with_radam']:
            rou_ = 2 / (1 - beta2) - 1
            rou_t = rou_ - 2 * t / (beta2 ** -t - 1)
            bias_c = 1 / (1 - beta1 ** t)
            if rou_t > group['radam_th']:
                bias_c *= math.sqrt(1 - beta2 ** t)
                bias_c *= math.sqrt(((rou_t - 4) * (rou_t - 2) * rou_) / ((rou_ - 4) * (rou_ - 2) * rou_t))
            else:
                eps_mode = 2
                bias_c = 0
            lr *= bias_c

        if self.opt_type == 'admax':
            eps_mode |= 0x10

        with torch.cuda.device(param.device.index):
            out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
            if out_p is None or out_p.dtype != grad.dtype:
                out_p = torch.tensor([], dtype=torch.float).to(param.data)

            weight_decay = group['weight_decay_rate']
            adamw(param.data,
                  out_p,
                  next_m,
                  next_v,
                  grad,
                  lr,
                  beta1,
                  beta2,
                  group['e'],
                  grad_scale,  # combined_scale, g = g/scale
                  state['step'],
                  eps_mode,  # self.eps_mode, esp inside sqrt:0, outside: 1, only update with momentum: 2
                  0,  # bias_correction,
                  weight_decay)

            out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
            if out_p is not None and out_p.dtype != grad.dtype:
                out_p.copy_(param.data)

    def get_state(self, param):
        state = self.state[param]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['next_m'] = torch.zeros_like(param.data)
            state['next_v'] = torch.zeros_like(param.data)
        return state


#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

""" FP16 optimizer wrapper
"""

from collections import defaultdict
import numpy as np
import math
import torch
import pdb
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import ctypes
# from ..utils import get_logger, boolean_string

# logger = get_logger()

__all__ = ['Fp16Optimizer', 'ExpLossScaler', 'get_world_size']


def get_world_size():
    try:
        wd = dist.get_world_size()
        return wd
    except:
        return 1


def fused_norm(input):
    return torch.norm(input, p=2, dtype=torch.float32)


class OptParameter(torch.Tensor):
    def __new__(cls, data, out_data=None, grad=None, name=None):
        param = torch.Tensor._make_subclass(cls, data)
        param._xgrad = grad
        param.out_data = out_data
        param._name = name
        return param

    @property
    def name(self):
        return self._name

    @property
    def grad(self):
        return self._xgrad

    @grad.setter
    def grad(self, grad):
        self._xgrad = grad


class Fp16Optimizer(object):
    def __init__(self, param_groups, optimizer_fn, loss_scaler=None, grad_clip_norm=1.0, lookahead_k=-1,
                 lookahead_alpha=0.5, rank=-1, distributed=False):
        # all parameters should on the same device
        groups = []
        original_groups = []
        self.rank = rank
        self.distributed = distributed
        if self.rank < 0:
            self.distributed = False
        for group in param_groups:
            if 'offset' not in group:
                group['offset'] = None
            if ('rank' not in group) or (not self.distributed):
                group['rank'] = -1
                assert group['offset'] is None, f"{group['names']}: {group['offset']}"
            group_rank = group['rank']
            params = group['params']  # parameter
            if len(params) > 1:
                flattened_params = _flatten_dense_tensors([p.data for p in params])
                unflattend_params = _unflatten_dense_tensors(flattened_params, [p.data for p in params])
                for uf, p in zip(unflattend_params, params):
                    p.data = uf
            else:
                flattened_params = params[0].data.view(-1)
                if group['offset'] is not None:
                    start, length = group['offset']
                    flattened_params = flattened_params.narrow(0, start, length)

            if params[0].dtype == torch.half:
                if self.rank == group_rank or (not self.distributed):
                    master_params = flattened_params.clone().to(torch.float).detach_().to(flattened_params.device)
                else:
                    master_params = flattened_params.clone().to(torch.float).detach_().cpu()
                group['params'] = [OptParameter(master_params, flattened_params, name='master')]
            else:
                group['params'] = [OptParameter(flattened_params, None, name='master')]

            o_group = defaultdict(list)
            o_group['names'] = group['names']
            o_group['params'] = params
            o_group['rank'] = group_rank
            o_group['offset'] = group['offset']

            group['names'] = ['master']

            original_groups.append(o_group)
            groups.append(group)
        self.param_groups = groups
        self.loss_scaler = loss_scaler
        self.optimizer = optimizer_fn(self.param_groups)
        self.original_param_groups = original_groups
        self.max_grad_norm = grad_clip_norm
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha

    def backward(self, loss):
        if self.loss_scaler:
            loss_scale, loss, step_loss = self.loss_scaler.scale(loss)
        else:
            loss_scale = 1
            step_loss = loss.item()

        loss.backward()
        return loss_scale, step_loss

    def step(self, lr_scale, loss_scale=1):
        grad_scale = self._grad_scale(loss_scale)
        if grad_scale is None or math.isinf(grad_scale):
            self.loss_scaler.update(False)
            return False

        if self.lookahead_k > 0:
            for p in self.param_groups:
                if 'la_count' not in p:
                    # init
                    # make old copy
                    p['la_count'] = 0
                    p['slow_params'] = [x.data.detach().clone().requires_grad_(False) for x in p['params']]
        self.optimizer.step(grad_scale, lr_scale)
        if self.lookahead_k > 0:
            for p in self.param_groups:
                p['la_count'] += 1
                if p['la_count'] == self.lookahead_k:
                    p['la_count'] = 0
                    for s, f in zip(p['slow_params'], p['params']):
                        s.mul_(1 - self.lookahead_alpha)
                        s.add_(f.data.detach() * self.lookahead_alpha)
                        f.data.copy_(s, non_blocking=True)
                        if hasattr(f, 'out_data') and f.out_data is not None:
                            f.out_data.copy_(f.data, non_blocking=True)

        if self.loss_scaler:
            self.loss_scaler.update(True)
        return True

    def zero_grad(self):
        for group, o_group in zip(self.param_groups, self.original_param_groups):
            for p in group['params']:
                p.grad = None
            for p in o_group['params']:
                p.grad = None

    def _grad_scale(self, loss_scale=1):
        named_params = {}
        named_grads = {}
        for g in self.original_param_groups:
            for n, p in zip(g['names'], g['params']):
                named_params[n] = p
                named_grads[n] = p.grad if p.grad is not None else torch.zeros_like(p.data)

        wd = get_world_size()

        def _reduce(group):
            grads = [named_grads[n] for n in group]
            if len(grads) > 1:
                flattened_grads = _flatten_dense_tensors(grads)
            else:
                flattened_grads = grads[0], view(-1)

            if wd > 1:
                flattened_grads /= wd
                handle = dist.all_reduce(flattened_grads, async_op=True)
            else:
                handle = None
            return flattened_grads, handle

        def _process_grad(group, flattened_grads, max_grad, norm):
            grads = [named_grads[n] for n in group]
            norm = norm.to(flattened_grads.device)
            norm = norm + fused_norm(flattened_grads) ** 2

            if len(grads) > 1:
                unflattend_grads = _unflatten_dense_tensors(flattened_grads, grads)
            else:
                unflattend_grads = [flattened_grads]

            for n, ug in zip(group, unflattend_grads):
                named_grads[n] = ug  # .to(named_params[n].data)

            return max_grad, norm

        group_size = 0
        group = []
        max_size = 32 * 1024 * 1024
        norm = torch.zeros(1, dtype=torch.float)
        max_grad = 0

        all_grads = []
        for name in sorted(named_params.keys(), key=lambda x: x.replace('deberta.', 'bert.')):
            group.append(name)
            group_size += named_params[name].data.numel()
            if group_size >= max_size:
                flatten, handle = _reduce(group)
                all_grads.append([handle, flatten, group])
                group = []
                group_size = 0
        if group_size > 0:
            flatten, handle = _reduce(group)
            all_grads.append([handle, flatten, group])
            group = []
            group_size = 0
        for h, fg, group in all_grads:
            if h is not None:
                h.wait()
            max_grad, norm = _process_grad(group, fg, max_grad, norm)

        norm = norm ** 0.5
        if torch.isnan(norm) or torch.isinf(norm):  # in ['-inf', 'inf', 'nan']:
            return None

        scaled_norm = norm.detach().item() / loss_scale
        grad_scale = loss_scale

        if self.max_grad_norm > 0:
            scale = norm / (loss_scale * self.max_grad_norm)
            if scale > 1:
                grad_scale *= scale

        for group, o_g in zip(self.param_groups, self.original_param_groups):
            grads = [named_grads[n] for n in o_g['names']]

            if len(grads) > 1:
                flattened_grads = _flatten_dense_tensors(grads)
            else:
                flattened_grads = grads[0].view(-1)
                if group['offset'] is not None:
                    start, length = group['offset']
                    flattened_grads = flattened_grads.narrow(0, start, length)
            if group['rank'] == self.rank or (not self.distributed):
                group['params'][0].grad = flattened_grads

        return grad_scale


class ExpLossScaler:
    def __init__(self, init_scale=2 ** 16, scale_interval=1000):
        self.cur_scale = init_scale
        self.scale_interval = scale_interval
        self.invalid_cnt = 0
        self.last_scale = 0
        self.steps = 0
        self.down_scale_smooth = 0

    def scale(self, loss):
        assert self.cur_scale > 0, self.init_scale
        step_loss = loss.float().detach().item()
        if step_loss != 0 and math.isfinite(step_loss):
            loss_scale = self.cur_scale
        else:
            loss_scale = 1
        loss = loss.float() * loss_scale
        return (loss_scale, loss, step_loss)

    def update(self, is_valid=True):
        if not is_valid:
            self.invalid_cnt += 1
            if self.invalid_cnt > self.down_scale_smooth:
                self.cur_scale /= 2
                self.cur_scale = max(self.cur_scale, 1)
                self.last_scale = self.steps
        else:
            self.invalid_cnt = 0
            if self.steps - self.last_scale > self.scale_interval:
                self.cur_scale *= 2
                self.last_scale = self.steps
        self.steps += 1

    def state_dict(self):
        state = defaultdict(float)
        state['steps'] = self.steps
        state['invalid_cnt'] = self.invalid_cnt
        state['cur_scale'] = self.cur_scale
        state['last_scale'] = self.last_scale
        return state

    def load_state_dict(self, state):
        self.steps = state['steps']
        self.invalid_cnt = state['invalid_cnt']
        self.cur_scale = state['cur_scale']
        self.last_scale = state['last_scale']