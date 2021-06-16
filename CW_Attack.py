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

# Load the datasets
# We randomly sample some images from the dataset.
def dataloader(img_size, data_dir, batch_size=1, num_workers=0, total_num=50000):
    # Setup the transformation
    img_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

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
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'fixed'], help="hyperparameter modes")
    parser.add_argument('--result_dir', type=str, default='result/CW', help="dir of the result")
    args = parser.parse_args()

    class_idx = json.load(open('./imagenet_class_index.json'))
    idx = [class_idx[str(k)][0] for k in range(len(class_idx))]
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    img_loader = dataloader(img_size=args.adv_size,
                            data_dir='/data/ImageNet/val',
                            batch_size=1, num_workers=0, total_num=50000)

        class Normalize(nn.Module):
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))

        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std


    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    device = torch.device('cuda:3')
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    if args.model_type == "inceptionv3":
        model = nn.Sequential(
            norm_layer, models.inception_v3(pretrained=True)).cuda()
    elif args.model_type == "vgg16":
        model = nn.Sequential(
            norm_layer, models.vgg16(pretrained=True)).cuda()
    elif args.model_type == "vgg19":
        model = nn.Sequential(
            norm_layer, models.vgg19(pretrained=True)).cuda()
    elif args.model_type == "resnet18":
        model = nn.Sequential(
            norm_layer, models.resnet18(pretrained=True)).cuda()
    elif args.model_type == "resnet50":
        model = nn.Sequential(
            norm_layer, models.resnet50(pretrained=True)).cuda()
    print("True Image & Predicted Label")

    if args.mode is 'fixed':
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
    else:
        if not os.path.isdir("./result/CWvgg19/test4types"):
        os.mkdir("./result/CWvgg19/test4types")
        if not os.path.isdir("./result/CWvgg19/test4types/weak"):
            os.mkdir("./result/CWvgg19/test4types/weak")
        if not os.path.isdir("./result/CWvgg19/test4types/strong"):
            os.mkdir("./result/CWvgg19/test4types/strong")
        if not os.path.isdir("./result/CWvgg19/test4types/medium"):
            os.mkdir("./result/CWvgg19/test4types/medium")
        if not os.path.isdir("./result/CWvgg19/test4types/out"):
            os.mkdir("./result/CWvgg19/test4types/out")
        if not os.path.isdir("./result/CWvgg19/test4types/clean"):
            os.mkdir("./result/CWvgg19/test4types/clean")
        path0 = os.path.join('./result/CWvgg19/test4types', 'CW_vgg19_log.csv')
        with open(path0, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["No.", "Prediction of Source image",
                 "Prediction of attacked image", 'Successful or not'])

    model.eval()

    correct = 0
    total = 0
    counts = 0
    i = 0
    torch.cuda.empty_cache()
    
    if args.mode is 'fixed':
        CW = CW.CW(model, c=10, kappa=0, steps=1000, lr=0.01)
        print('fixed attack')
        if args.attack_goal == 'T':
        CW.set_attack_mode("targeted") 
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
    
    else:
        print('random attack')
            for images, l in img_loader:

        # CW
        c = np.random.randint(1, 101)

        kappa_weak = np.random.randint(0, 5)
        kappa_medium = np.random.randint(5, 15)
        kappa_strong = np.random.randint(15, 21)
        kappa_out = np.random.randint(21, 30)

        steps = np.random.randint(100, 1001)

        CW_weak = CW.CW(model, c=c, kappa=kappa_weak, steps=steps, lr=0.01)
        CW_medium = CW.CW(model, c=c, kappa=kappa_medium, steps=steps, lr=0.01)
        CW_strong = CW.CW(model, c=c, kappa=kappa_strong, steps=steps, lr=0.01)
        CW_out = CW.CW(model, c=c, kappa=kappa_out, steps=steps, lr=0.01)

        print('C:', c)
        print('kappa_weak:', kappa_weak)
        print('kappa_medium:', kappa_medium)
        print('kappa_strong:', kappa_strong)
        print('kappa_out:', kappa_out)
        print('Steps:', steps)

        print('No.', counts)

        images = images.cuda()
        pred = model(images)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().detach().numpy().flatten()
        labels = np.argmax(pred)
        pred_class = idx2label[labels]
        labels = torch.tensor([labels])
        print('Pred of Source:', pred_class)
        print('labels ', labels)

        # CW_weak
        adv_images_weak = CW_weak(images, labels)
        adv_images_weak = adv_images_weak.cuda()

        adv_outputs_weak = model(adv_images_weak)

        adv_pred_weak = torch.softmax(adv_outputs_weak, dim=1)
        adv_pred_weak = adv_pred_weak.data.cpu().detach().numpy().flatten()
        adv_pred_label_weak = np.argmax(adv_pred_weak)
        adv_pred_class_weak = idx2label[int(adv_pred_label_weak)]
        print('CW_Adv_weak:', adv_pred_class_weak)

        # CW_medium
        adv_images_medium = CW_medium(images, labels)
        adv_images_medium = adv_images_medium.cuda()

        adv_outputs_medium = model(adv_images_medium)

        adv_pred_medium = torch.softmax(adv_outputs_medium, dim=1)
        adv_pred_medium = adv_pred_medium.data.cpu().detach().numpy().flatten()
        adv_pred_label_medium = np.argmax(adv_pred_medium)
        adv_pred_class_medium = idx2label[int(adv_pred_label_medium)]
        print('CW_Adv_medium:', adv_pred_class_medium)

        # CW_strong
        adv_images_strong = CW_strong(images, labels)
        adv_images_strong = adv_images_strong.cuda()

        adv_outputs_strong = model(adv_images_strong)

        adv_pred_strong = torch.softmax(adv_outputs_strong, dim=1)
        adv_pred_strong = adv_pred_strong.data.cpu().detach().numpy().flatten()
        adv_pred_label_strong = np.argmax(adv_pred_strong)
        adv_pred_class_strong = idx2label[int(adv_pred_label_strong)]
        print('CW_Adv_strong:', adv_pred_class_strong)

        # CW_out
        adv_images_out = CW_out(images, labels)
        adv_images_out = adv_images_out.cuda()

        adv_outputs_out = model(adv_images_out)

        adv_pred_out = torch.softmax(adv_outputs_out, dim=1)
        adv_pred_out = adv_pred_out.data.cpu().detach().numpy().flatten()
        adv_pred_label_out = np.argmax(adv_pred_out)
        adv_pred_class_out = idx2label[int(adv_pred_label_out)]
        print('CW_Adv_out:', adv_pred_class_out)

        if adv_pred_class_weak is not pred_class and adv_pred_class_medium is not pred_class and adv_pred_class_strong is not pred_class and adv_pred_class_out is not pred_class:
            status = 'S'
            counts += 1

            # CW_weak
            im_CW_weak = adv_images_weak.squeeze(0).data.cpu().detach().numpy()
            im_CW_weak = (np.transpose(im_CW_weak, (1, 2, 0)) * 255).astype(np.uint8)
            im_CW_weak = Image.fromarray(im_CW_weak)

            # CW_medium
            im_CW_medium = adv_images_medium.squeeze(0).data.cpu().detach().numpy()
            im_CW_medium = (np.transpose(im_CW_medium, (1, 2, 0)) * 255).astype(np.uint8)
            im_CW_medium = Image.fromarray(im_CW_medium)

            # CW_strong
            im_CW_strong = adv_images_strong.squeeze(0).data.cpu().detach().numpy()
            im_CW_strong = (np.transpose(im_CW_strong, (1, 2, 0)) * 255).astype(np.uint8)
            im_CW_strong = Image.fromarray(im_CW_strong)

            # CW_out
            im_CW_out = adv_images_out.squeeze(0).data.cpu().detach().numpy()
            im_CW_out = (np.transpose(im_CW_out, (1, 2, 0)) * 255).astype(np.uint8)
            im_CW_out = Image.fromarray(im_CW_out)

            # clean
            im = images.squeeze(0).data.cpu().detach().numpy()
            im = (np.transpose(im, (1, 2, 0)) * 255).astype(np.uint8)
            im = Image.fromarray(im)

            im_CW_weak.save(
                './result/CWvgg19/test4types/weak/{}_{}_{}_{}_{}_{}_{}.bmp'.format(args.task,
                                                                                            'CW',
                                                                                            args.attack_goal,
                                                                                            args.model_type,
                                                                                            args.strategy,
                                                                                            counts,
                                                                                            status))

            im_CW_medium.save(
                './result/CWvgg19/test4types/medium/{}_{}_{}_{}_{}_{}_{}.bmp'.format(args.task,
                                                                                              'CW',
                                                                                              args.attack_goal,
                                                                                              args.model_type,
                                                                                              args.strategy,
                                                                                              counts,
                                                                                              status))

            im_CW_strong.save(
                './result/CWvgg19/test4types/strong/{}_{}_{}_{}_{}_{}_{}.bmp'.format(args.task,
                                                                                              'CW',
                                                                                              args.attack_goal,
                                                                                              args.model_type,
                                                                                              args.strategy,
                                                                                              counts,
                                                                                              status))

            im_CW_out.save(
                './result/CWvgg19/test4types/out/{}_{}_{}_{}_{}_{}_{}.bmp'.format(args.task,
                                                                                           'CW',
                                                                                           args.attack_goal,
                                                                                           args.model_type,
                                                                                           args.strategy,
                                                                                           counts,
                                                                                           status))
            im.save(
                './result/CWvgg19/test4types/clean/{}_{}_{}.bmp'.format('Clean',
                                                                              'CW',
                                                                              counts, ))

            with open(path0, 'a', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(
                    [counts, pred_class,
                     [adv_pred_class_weak, adv_pred_class_medium, adv_pred_class_strong, adv_pred_class_out],
                     status])
    
