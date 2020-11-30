import numpy as np
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import sys
import argparse
import os
from PIL import Image
from Lpnorm_attacks import CW


# def image_folder_custom_label(root, transform, custom_label):
#     # custom_label
#     # type : List
#     # index -> label
#     # ex) [ 'goldfish', 'giant_panda', 'tiger_shark']
#
#     old_data = dsets.ImageFolder(root=root, transform=transform)
#     old_classes = old_data.classes
#
#     label2idx = {}
#
#     for i, item in enumerate(idx2label):
#         label2idx[item] = i
#
#     new_data = dsets.ImageFolder(root=root, transform=transform,
#                                  target_transform=lambda x: custom_label.index(old_classes[x]))
#     new_data.classes = idx2label
#     new_data.class_to_idx = label2idx
#
#     return new_data
# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(img_size, data_dir, batch_size=1, num_workers=0, total_num=50000):
    # Setup the transformation
    img_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # img_transforms = transforms.Compose([
    #     transforms.Resize(size=(224, 224)),
    #     transforms.ToTensor(),
    # ])

    index = np.arange(total_num)
    np.random.shuffle(index)
    img_index = index[:img_size]
    print('inedx:', img_index)

    img_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=img_transforms)

    img_loader = DataLoader(dataset=img_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(img_index),
                            num_workers=num_workers, pin_memory=True, shuffle=False)

    return img_loader




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                       'Face Recognition')
    parser.add_argument(
        '--attack_type', type=str, default='PGD', choices=['FGSM', 'CW', 'PGD', 'Patch', 'trojan']
    )
    parser.add_argument(
        '--attack_goal', type=str, default='N', choices=['N', 'T'], help='Non-target, Target'
    )
    parser.add_argument('--attack_target', type=int, default=None, help="target attack label")
    parser.add_argument("--model_type", type=str, default='resnet18',
                        choices=['inceptionv3', 'resnet18', 'resnet50', 'vgg16', 'vgg19'],
                        help='pre-trained target classifier')
    parser.add_argument(
        "--strategy", type=str, default='WB', choices=['WB', 'TBB', 'QBB'], help='White Box, Transfer based black box, '
                                                                                 'Query based balck box '
    )
    parser.add_argument('--adv_size', type=int, default=50, help="number of generating adversarial images")
    parser.add_argument('--result_dir', type=str, default='result/CW', help="dir of the result")
    args = parser.parse_args()

    class_idx = json.load(open('/home/liyize/Codes/Adversarial_Patch_Attack/imagenet_class_index.json'))
    idx = [class_idx[str(k)][0] for k in range(len(class_idx))]
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    ])

    img_loader = dataloader(img_size=args.adv_size,
                            data_dir='/data/ImageNet/val',
                            batch_size=1, num_workers=0, total_num=50000)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    device = torch.device('cuda:3')
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    if args.model_type == "inceptionv3":
        model = models.inception_v3(pretrained=True).cuda()
    elif args.model_type == "vgg16":
        model = models.vgg16(pretrained=True).cuda()
    elif args.model_type == "vgg19":
        model = models.vgg19(pretrained=True).cuda()
    elif args.model_type == "resnet18":
        model = models.resnet18(pretrained=True).cuda()
    elif args.model_type == "resnet50":
        model = models.resnet50(pretrained=True).cuda()
    print("True Image & Predicted Label")

    if not os.path.isdir("result"):
        os.mkdir("result")
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    path = os.path.join(args.result_dir, 'CW_log.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["No.", "Prediction of Source image", "Confidence %", "Source image under attack label ", "Confidence %",
             "Attacked image under source label ", "Confidence %",
                                                   "Prediction of attacked image", "Confidence %", 'Successful or not'])

    model.eval()

    correct = 0
    total = 0
    counts = 0
    i = 0
    torch.cuda.empty_cache()

    CW = CW.CW(model, c=10, kappa=0, steps=1000, lr=0.01)
    if args.attack_goal == 'T':
        CW.set_attack_mode("targeted")
    # for images, labels in n_loader:
    for images, labels in img_loader:
        print('No.', i)
        print('Label0:', idx2label[labels])
        if args.attack_goal == 'T':
            labels = torch.tensor([args.attack_target])
        # for images, labels in normal_loader:
        images = images.cuda()
        pred = model(images)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().detach().numpy().flatten()
        pred_label = np.argmax(pred)
        pred_class = idx2label[pred_label]
        pred_prob = pred[pred_label]
        print('OG:', pred_class)
        print('OG %:', 100 * round(pred_prob, 6))
        if pred_label is not labels:
            print('Wrong Prediction')

        adv_images = CW(images, labels)

        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])

        im = adv_images.squeeze(0).data.cpu().float().detach().numpy()
        # im = (np.transpose((im * STD + MEAN) *255), (1, 2, 0)).astype(np.uint8)
        im = ((np.transpose(im, (1, 2, 0)) * STD + MEAN) * 255).astype(np.uint8)
        im = Image.fromarray(im)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        adv_images1 = preprocess(im)
        adv_images1 = adv_images1.unsqueeze(0).float()
        adv_images1 = adv_images1.cuda()

        outputs = model(adv_images1)

        pred1 = torch.softmax(outputs, dim=1)
        pred1 = pred1.data.cpu().detach().numpy().flatten()
        pred1_label = np.argmax(pred1)
        pred1_class = idx2label[int(pred1_label)]
        pred1_prob = pred1[pred1_label]

        pred2_prob = pred[pred1_label]
        pred3_prob = pred1[pred_label]

        print('OG_Adv:', pred1_class)
        print('OG_Adv%:', pred2_prob)

        print('Adv_OG:', pred_class)
        print('Adv_OG%:', pred3_prob)

        print('Adv:', pred1_class)
        print('Adv%:', pred1_prob)
        total += 1

        # MEAN = np.array([0.485, 0.456, 0.406])
        # STD = np.array([0.229, 0.224, 0.225])
        # im = adv_images.squeeze(0).data.cpu().detach().numpy()
        # im = ((np.transpose(im, (1, 2, 0)) * STD + MEAN) * 255).astype(np.uint8)
        # im = Image.fromarray(im)
        if args.attack_goal == 'N':
            if pred1_class is not pred_class:
                status = 'S'
                counts += 1
            elif pred1_class is pred_class:
                if pred1_label == labels:
                    status = 'Uns1'
                else:
                    print('Wrong pred')
                    status = 'Uns0'
                    counts += 1

        elif args.attack_goal == 'T':
            if pred1_label == args.attack_target:
                status = 'S'
                counts += 1
            else:
                status = 'Uns'

        im.save('/home/liyize/Codes/AdvTool/result/CW/{}_{}_{}_{}_{}_{}_{}.jpg'.format(args.task,
                                                                                        args.attack_type,
                                                                                        args.attack_goal,
                                                                                        args.model_type,
                                                                                        args.strategy, i, status))
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(
                [i, pred_class, round(pred_prob * 100, 1), pred1_class, round(pred2_prob * 100, 1),
                 pred_class, round(pred3_prob * 100, 1), pred1_class, round(pred1_prob * 100, 1), status])
        i = i + 1

    print('Generate Success accuracy: %.2f %%' % (100 * float(counts) / total))
