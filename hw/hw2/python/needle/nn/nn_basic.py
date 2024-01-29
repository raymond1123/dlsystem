"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        bias = self.bias.reshape((1, self.weight.shape[1])).broadcast_to((X.shape[0], self.out_features))
        return X@self.weight + bias
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        mul = 1
        for i,s in enumerate(shape):
            if i>0:
                mul*=s

        new_shape = [shape[0], mul]
        return ops.reshape(X, new_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_samples, _ = logits.shape
        n = logits.shape[1]

        y_one_hot = init.one_hot(n, y, dtype=y.dtype, 
                                 requires_grad=y.requires_grad)

        zy = ops.summation(logits*y_one_hot, axes=1)

        loss = ops.summation(ops.logsumexp(logits, axes=1) - zy)
        loss /= Tensor(num_samples, dtype=loss.dtype)

        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION

        self.weight = Parameter(init.ones(dim, device=device, 
                                dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, 
                               dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(dim, device=device, 
                               dtype=dtype, requires_grad=True)
        self.running_var = init.ones(dim, device=device, 
                                dtype=dtype, requires_grad=True)

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_feat = x.shape
        if self.training:
            expect_obs = ops.summation(x, axes=0)/batch_size
            expect = ops.broadcast_to(ops.reshape(expect_obs, (1, num_feat)), (x.shape))
            var_obs = ops.summation(ops.power_scalar((x-expect), 2)/batch_size, axes=0)
            var = ops.broadcast_to(ops.reshape(var_obs, (1, num_feat)), (x.shape))

            ## update runing mean and var
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*expect_obs.data
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var_obs.data

            weight = ops.broadcast_to(ops.reshape(self.weight, (1, num_feat)), (x.shape))
            bias = ops.broadcast_to(ops.reshape(self.bias, (1, num_feat)), (x.shape))

            var = ops.power_scalar(var+self.eps, 0.5)
            x = (x - expect)/var*weight+bias
        else:
            expect = ops.broadcast_to(ops.reshape(self.running_mean, (1, num_feat)), (x.shape))
            var = ops.broadcast_to(ops.reshape(self.running_var, (1, num_feat)), (x.shape))

            weight = ops.broadcast_to(ops.reshape(self.weight, (1, num_feat)), (x.shape))
            bias = ops.broadcast_to(ops.reshape(self.bias, (1, num_feat)), (x.shape))

            var = ops.power_scalar(var+self.eps, 0.5)
            x = weight*(x - expect)/var+bias

        return x
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, 
                                dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, 
                               dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_feat = x.shape
        expect_obs = ops.summation(x, axes=1)/num_feat
        expect = ops.broadcast_to(ops.reshape(expect_obs, (batch_size, 1)), (x.shape))
        var_obs = ops.summation(ops.power_scalar((x-expect), 2)/num_feat, axes=1)
        var = ops.broadcast_to(ops.reshape(var_obs, (batch_size, 1)), (x.shape))

        weight = ops.broadcast_to(ops.reshape(self.weight, (1, num_feat)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, num_feat)), x.shape)

        var = ops.power_scalar(var+self.eps, 0.5)
        x = (x - expect)/var*weight+bias

        return x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, dtype="float32")/(1-self.p)
            return x*mask

        else:
            return x

        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
