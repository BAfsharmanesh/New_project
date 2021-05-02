import random
import os
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_timestamp():
    import time
    timestamp = ''
    for i, d in enumerate(time.localtime()):
        if i == 3:
            d += 8
        timestamp += str(d) + '-'
        if i == 4:
            break
    return timestamp[:-1]

def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt((xhat-x)**2 + (yhat-y)**2) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]
