import copy
import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms


class WhiteBoxAttack(object):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, input_size=224, epsilon=32, alpha=5,
                 num_iters=100, early_stopping=None, use_cuda=False):
        '''__INIT__
            reference:
            Kurakin A, Goodfellow I, Bengio S.
            Adversarial examples in the physical world[J].
            arXiv preprint arXiv:1607.02533, 2016.

            model: model instance or list of model instances
            input_size: int, size of input tentor to model
            epsilon: int, limit on the perturbation size
            alpha: int, step size for gradient-based attack
            num_iters: int, number of iterations
            early_stopping: int ot None, attack will not stop unless loss stops improving
            use_cuda: bool, True or False, whether to use GPU
        '''

        self.alpha = alpha / 255
        self.num_iters = num_iters
        self.epsilon = epsilon / 255
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        if not isinstance(model, list):
            model = [model]
        model = [copy.deepcopy(m) for m in model]
        for m in model:
            m.eval()
            if self.use_cuda:
                m.cuda()
        self.model = model

        self.l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        return

    def __call__(self, image_path, label, norm=None, target=False):
        '''__CALL__
            image_path: string, path of input image
            label: int, the true label of input image if target is False,
                   the target label to learn if target is True
            target: bool, if True, perform target adversarial attack;
                    if False, perform non-target adversarial attack
        '''

        self.target = target
        self.norm = norm
        src_image = Image.open(image_path)
        adv_image = self.forward(src_image, label)
        return adv_image

    def forward(self, src_image, label):
        image = self.preprocess(src_image).unsqueeze(0)
        origin = image.clone().detach()
        label = torch.LongTensor([label])
        if self.use_cuda:
            image, origin, label = image.cuda(), origin.cuda(), label.cuda()

        num_no_improve, best_iter = 0, None
        best_loss, best_adv_image = None, None
        for i in range(self.num_iters):
            image.requires_grad = True
            pred = [m(image) for m in self.model]
            loss = self.__loss(pred, label, image, origin)
            # loss.backward(retain_graph=True)

            iter_msg = '[Running]-[Step:{:0=3d}/{:0=3d}]-[Loss:{:12.6f}]'
            # print(iter_msg.format(i + 1, self.num_iters, loss.item()), end='\r')

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = image.clone()
                best_loss = loss.item()
                num_no_improve = 0
                best_iter = i
            else:
                num_no_improve += 1
            if self.__stop(num_no_improve):
                break

            image = self.__PGD(loss, image, origin)
            image_clamp = self.__clamp(image)
            image = image_clamp.detach()

        stop_msg = '\n[Stopped]-[Step:{:0=3d}]-[Loss:{:.6f}]'
        # print(stop_msg.format(best_iter + 1, best_loss))

        adv_image = self.__post_process(best_adv_image)
        return adv_image

    def __loss(self, pred, label, image, origin):
        def compute_ce_loss(p, l):
            if self.target:
                ce_loss = self.ce_loss(p, l)

            else:
                # ce_loss = 1 / torch.clamp(self.ce_loss(p, l), min=1e-8)
                ce_loss = -self.ce_loss(p, l)
            return ce_loss

        ce_loss = 0
        for p in pred:
            ce_loss += compute_ce_loss(p, label)
        ce_loss /= len(pred)

        return ce_loss

    def __stop(self, num_no_improve):
        return (self.early_stopping is not None) and \
               (num_no_improve == self.early_stopping)

    def __PGD(self, loss, image, origin):
        grad = torch.autograd.grad(loss, image, retain_graph=False)[0]

        # print(perturbation.size())

        # def l2_project(X, r):
        #     '''project data X onto l2 ball of radius r.'''
        #     n = X.shape[0]
        #     norms = X.data.view(n, -1).norm(dim=1).view(n, 1, 1, 1)
        #     X.data *= norms.clamp(0., r) / norms
        #     return X
        def normalize_by_pnorm(x, p=2, small_constant=self.epsilon):
            """
            Normalize gradients for gradient (not gradient sign) attacks.
            # TODO: move this function to utils
            :param x: tensor containing the gradients on the input.
            :param p: (optional) order of the norm for the normalization (1 or 2).
            :param small_constant: (optional float) to avoid dividing by zero.
            :return: normalized gradients.
            """
            # loss is averaged over the batch so need to multiply the batch
            # size to find the actual gradient of each input sample

            assert isinstance(p, float) or isinstance(p, int)
            x = x.reshape(-1, 224)
            norm = np.linalg.norm(x, ord=2)
            norm = max(norm, small_constant)
            x_norm = 1. / norm * x
            return x_norm.reshape(3, 224, 224)

        if self.norm == 'PGD_linf':
            # print('linf')
            perturbation = image - self.alpha * grad.sign() - origin
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            # print('P:{}'.format(perturbation - perturbation0))
            image = origin + perturbation

        elif self.norm == 'PGD_l2':
            # print('l2')
            perturbation = image - self.alpha * grad.sign() - origin
            perturbation = normalize_by_pnorm(perturbation.cpu().detach(), p=2, small_constant=self.epsilon )
            image = origin + perturbation.cuda()

        # image = origin - perturbation

        return image

    def __clamp(self, image):
        image_clamp = []
        for i in range(image.size()[1]):
            cmin, cmax = self.RANGE[i]
            image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
            image_clamp.append(image_i)
        image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
        return image_clamp

    def __post_process(self, best_adv_image):
        adv_image = best_adv_image.squeeze(0).data.cpu().detach().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = adv_image * self.STD + self.MEAN
        adv_image = np.round(adv_image * 255.0).astype(np.uint8)
        return adv_image
