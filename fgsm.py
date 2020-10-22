import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def FGSMAttack(model, images, labels, epsilon):
    images = images.to(device)
    labels = labels.to(device)

    X = np.copy(images.numpy())

    X_var = to_var(torch.from_numpy(X), requires_grad=True)

    y_var = to_var(torch.LongTensor(labels))

    scores = model(X_var)
    loss = self.loss_fn(scores, y_var)
    loss.backward()
    grad_sign = X_var.grad.data.cpu().sign().numpy()

    X += self.epsilon * grad_sign
    X = np.clip(X, 0, 1)

    adv_images = torch.from_numpy(X)
    return adv_images

