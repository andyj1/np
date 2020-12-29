import torch
from torch.optim import *


def check_optimizer(optimizer_type, modify_grad=False):
    testvar = torch.ones([])
    testvar.requires_grad = True

    if optimizer_type is SGD:
        opt = optimizer_type([testvar], 1e-3)
    else:
        opt = optimizer_type([testvar])

    def closure():
        if modify_grad:
            opt.zero_grad()
        if modify_grad:
            testvar.backward()
        return testvar

    for i in range(1000):
        opt.step(closure)

    return testvar.item()


optimizers = [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD]

for optimizer_type in optimizers:
    opt_name = optimizer_type.__name__
    print(opt_name.ljust(10), "not modifying grad:", check_optimizer(optimizer_type, False))
    print(opt_name.ljust(10), "    modifying grad:", check_optimizer(optimizer_type, True))