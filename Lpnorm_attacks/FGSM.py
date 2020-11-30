import torch
import torch.nn as nn
import numpy as np

from baselpnormattack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
         attack = torchattacks.FGSM(model, eps=0.007)
         adv_images = attack(images, labels)

    """
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps

    def __clamp(self, image):
        image_clamp = []
        for i in range(image.size()[1]):
            cmin, cmax = self.RANGE[i]
            image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
            image_clamp.append(image_i)
        image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
        return image_clamp

    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        loss = self._targeted * loss(outputs, labels).to(self.device)

        grad = torch.autograd.grad(loss, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        image_clamp = self.__clamp(adv_images)
        adv_images = image_clamp.detach()

        return adv_images