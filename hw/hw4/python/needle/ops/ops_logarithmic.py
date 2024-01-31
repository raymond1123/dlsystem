from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(self.axes)
        b_shape = list(Z.shape)

        if self.axes!=None:
            if isinstance(self.axes, tuple):
                for i in self.axes:
                    b_shape[i] = 1
            if isinstance(self.axes, int):
                    b_shape[self.axes] = 1
        else:
            b_shape = [1]*len(Z.shape)

        max_bz = max_z.reshape(b_shape).broadcast_to(Z.shape)
        log_sum_exp_z = (Z-max_bz).exp().sum(self.axes).log()

        return log_sum_exp_z + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes

        if isinstance(self.axes, tuple):
            for axis in axes:
                expand_shape[axis] = 1
        if isinstance(self.axes, int):
                expand_shape[self.axes] = 1

        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

