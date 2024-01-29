"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        #self.clip_grad_norm()

        for w in self.params:
            if w not in self.u: self.u[w] = 0.
            grad = w.grad.data + self.weight_decay * w.data
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad
            ### call @property-->data() and @data.setter-->data()
            w.data = w.data - self.lr * self.u[w]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        max_g = np.sqrt(max_norm)
        for w in self.params:
            cached_data = w.grad.cached_data
            cached_data[cached_data>max_g]=max_g
            w.grad.cached_data = cached_data

        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if w not in self.m: self.m[w] = 0.
            if w not in self.v: self.v[w] = 0.

            grad = w.grad.data + self.weight_decay * w.data

            self.m[w] = self.beta1* self.m[w] + (1 - self.beta1) * grad
            unbiased_m = self.m[w] / (1-self.beta1**self.t)

            self.v[w] = self.beta2* self.v[w] + (1 - self.beta2) * (grad**2)
            unbiased_v = self.v[w] / (1-self.beta2**self.t)

            w.data = w.data - self.lr * unbiased_m/(unbiased_v**0.5+self.eps)
        ### END YOUR SOLUTION

