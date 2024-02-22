from torch import optim
import torch
import math


class LRPolicy(object):
    def __init__(self, powr):
        self.powr = powr

    def __call__(self, iter):
        return math.pow(1-iter/102600, self.powr)



def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    #optimizer = optim.Adam(params, lr=lr)
    # step_size = int(epochs * .8)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy(powr=0.9))
    # print("Step size: %d" % step_size)
    return optimizer, scheduler
