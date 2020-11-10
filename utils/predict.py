import copy
import torch
import numpy as np

from PIL import Image
from torchvision import transforms


class AttackPredict(object):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, input_size=224, class_label=None, use_cuda=False):

        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.model = copy.deepcopy(model).eval()
        if self.use_cuda:
            self.model.cuda()

        self.class_label = class_label
        return

    def run(self, image_path, label2=None):
        image = self.preprocess(Image.open(image_path))

        image = image.unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()

        pred = self.model(image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().detach().numpy().flatten()
        pred_label = np.argmax(pred)
        pred_prob = pred[pred_label]
        other_prob = pred[label2]

        if self.class_label is not None:
            if label2 is not None:
                pred_class = self.class_label[pred_label]
                other_class = self.class_label[label2]
                return pred_label, pred_prob, pred_class, label2, other_prob, other_class
            else:
                pred_class = self.class_label[pred_label]
                return pred_label, pred_prob, pred_class

        # return pred_label, pred_prob
