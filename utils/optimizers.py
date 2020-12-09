from utils.constants import Cte
import torch
def get_optimizer(name):
    if name == Cte.ADAM:
        return torch.optim.Adam
    elif name == Cte.ADAGRAD:
        return torch.optim.Adagrad
    elif name == Cte.ADADELTA:
        return torch.optim.Adadelta
    elif name == Cte.RMS:
        return torch.optim.RMSprop
    elif name == Cte.ASGD:
        return torch.optim.ASGD
    else:
        raise NotImplementedError


def get_scheduler(name):
    if name == Cte.STEP_LR:
        return torch.optim.lr_scheduler.StepLR
    else:
        raise NotImplementedError
