import torch
import torch.nn as nn
import numpy as np

from baselpnormattack import Attack


class PGD(Attack):
    """
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        attack = torchattacks.PGD(model, eps = 8/255, alpha = 1/255, steps=40, random_start=False)
        adv_images = attack(images, labels)

    """
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    # def __clamp(self, image):
    #     image_clamp = []
    #     for i in range(image.size()[1]):
    #         cmin, cmax = self.RANGE[i]
    #         image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
    #         image_clamp.append(image_i)
    #     image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
    #     return image_clamp

    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # image_clamp = self.__clamp(images + delta)
            # adv_images = image_clamp.detach()

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            # images = torch.clamp(ori_images + eta, min=torch.min(images.data), max=torch.max(images.data)).detach_()

        return adv_images