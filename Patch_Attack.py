# Adversarial Patch Attack

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.utils

import argparse
import csv
import os
import numpy as np
from PIL import Image
# import cv2

from Patch_utils.patch_utils import *
from Patch_utils.utils import *


# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy

def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=1000):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
        (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = -torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = -lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                      applied_patch.type(torch.FloatTensor)) + torch.mul(
            (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch


# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    i = 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                          applied_patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
                i = i + 1

    # print(i)
    # print('test_success:', test_success)
    # print('test_actual_total:', test_actual_total)

    return test_success / test_actual_total


# Generate adversarial images with the patch on dataset
def advimg_patch(patch_type, target, patch, adv_loader, model, model_type, task, attack_type, attack_goal, path0,
                 class_label, strategy):
    model.eval()
    adv_total, adv_actual_total, adv_success = 0, 0, 0
    i = 0
    for (image, label) in adv_loader:
        adv_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        pred = torch.softmax(output, dim=1)
        pred = pred.data.cpu().detach().numpy().flatten()
        pred_label = np.argmax(pred)
        pred_class = class_label[pred_label]
        pred_prob = pred[pred_label]
        # print('OG:', pred_class)
        # print('OG %:', round(100 * pred_prob, 1))

        # if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
        #     adv_actual_total += 1
        applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                      applied_patch.type(torch.FloatTensor)) + torch.mul(
            (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = perturbated_image.cuda()
        output1 = model(perturbated_image)
        _, predicted1 = torch.max(output1.data, 1)
        pred1 = torch.softmax(output1, dim=1)
        pred1 = pred1.data.cpu().detach().numpy().flatten()
        pred1_label = np.argmax(pred1)
        pred1_class = class_label[pred1_label]
        pred1_prob = pred1[pred1_label]

        pred2_prob = pred[pred1_label]
        pred3_prob = pred1[pred_label]

        # print('OG_Adv:', pred1_class)
        # print('OG_Adv%:', pred2_prob)
        #
        # print('Adv_OG:', pred_class)
        # print('Adv_OG%:', pred3_prob)
        #
        # print('Adv:', pred1_class)
        # print('Adv%:', pred1_prob)

        adv_success += 1
        perturbated_image = perturbated_image.squeeze(0).data.cpu().detach().numpy()
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        im = ((np.transpose(perturbated_image, (1, 2, 0)) * std + mean) * 255).astype(np.uint8)
        im = Image.fromarray(im)

        if predicted1[0].data.cpu().numpy() == target:
            status = 'S'
        else:
            status = 'Uns'
        with open(path0, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [i, pred_class, round(pred_prob * 100, 6), pred1_class, round(pred2_prob * 100, 6),
                 pred_class, round(pred3_prob * 100, 6), pred1_class, round(pred1_prob * 100, 6), status])
        # im.save('E:/Academy/LAB/Diagnosis for Adversaries/Adversarial_Patch_Attack-master/adv_img/{}_{}_{}{}_{}.jpg'.format(area,
        #                                                                                      attack_type, attack_goal,target, i))
        im.save('/home/liyize/Codes/AdvTool/result/Patch/{}_{}_{}_{}_{}_{}_{}.jpg'.format(
            task,
            attack_type, attack_goal, model_type, strategy, i, status))
        i = i + 1

    # print('Adv_pic number :', i)
    # print('adv_success:', adv_success)
    # print('adv_actual_total:', adv_actual_total)
    #
    # return adv_success / adv_actual_total


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                       'Face Recognition')
    parser.add_argument('--attack_type', type=str, default='Patch', choices=['Patch'])
    parser.add_argument('--attack_goal', type=str, default='T', help='Target')
    parser.add_argument("--model_type", type=str, default='resnet18',
                        choices=['inceptionv3', 'resnet18', 'resnet50', 'vgg16', 'vgg19'],
                        help='pre-trained target classifier')
    parser.add_argument(
        "--strategy", type=str, default='WB', choices=['WB', 'TBB', 'QBB'], help='White Box, Transfer based black box, '
                                                                                 'Query based balck box '
    )
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
    parser.add_argument('--train_size', type=int, default=1000, help="number of training images")
    parser.add_argument('--test_size', type=int, default=600, help="number of test images")
    parser.add_argument('--adv_size', type=int, default=50, help="number of generating adversarial images")
    parser.add_argument('--noise_percentage', type=float, default=0.06,
                        help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.9,
                        help="Stop attack on image when target classifier reaches this value for target class")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=500, help="max iteration")
    parser.add_argument('--attack_target', type=int, default=859, help="target label")
    parser.add_argument('--epochs', type=int, default=8, help="total epoch")
    # parser.add_argument('--data_dir', type=str,
    #                     default='E:/Academy/LAB/Diagnosis for Adversaries/Adversarial_Patch_Attack-master/ImageNet/val',
    #                     help="dir of the dataset")
    parser.add_argument('--data_dir', type=str, default='/data/ImageNet/val',
                        help="dir of the dataset")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--GPU', type=str, default='1', help="index pf used GPU")
    parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
    parser.add_argument('--result_dir', type=str, default='result/Patch', help="dir of the dataset")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    # device = torch.device('cuda:1')

    # Load label-class pairs of ImageNet
    # class_label_dict = json.load(open('E:/Academy/LAB/Diagnosis for '
    #                                   'Adversaries/AdversarialAttack-master/data/imagenet_class_index.json'))
    class_label_dict = json.load(open('/home/liyize/Codes/Adversarial_Patch_Attack/imagenet_class_index.json'))
    class_label = [class_label_dict[str(k)][1] for k in range(len(class_label_dict))]

    # Load the model
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

    model.eval()

    # Load the datasets
    train_loader, test_loader, adv_loader = dataloader(args.train_size, args.test_size, args.adv_size, args.data_dir,
                                                       args.batch_size,
                                                       args.num_workers, 50000)

    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100 * trainset_acc,
                                                                                              100 * test_acc))

    # Initialize the patch
    patch = patch_initialization(args.patch_type, image_size=(3, 224, 224), noise_percentage=args.noise_percentage)
    print('The shape of the patch is', patch.shape)

    if not os.path.isdir("result"):
        os.mkdir("result")
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.isdir('result/Patch/training_results'):
        os.mkdir('result/Patch/training_results')

    path0 = os.path.join(args.result_dir, 'Patch_log.csv')
    path1 = os.path.join('result/Patch/training_results', 'Patch_train_test_log.csv')

    with open(path0, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["No.", "Prediction of Source image", "Confidence %", "Source image under attack label ", "Confidence %",
             "Attacked image under source label ", "Confidence %",
             "Prediction of attacked image", "Confidence %", 'Successful or not'])

    with open(path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate, best_patch_success_test_rate = 0, 0, 0
    best_patch = np.zeros(patch.shape)

    # Generate the patch
    for epoch in range(args.epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] == label and predicted[0].data.cpu().numpy() != args.attack_target:
                train_actual_total += 1
                applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch,
                                                                              image_size=(3, 224, 224))
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.attack_target,
                                                                args.probability_threshold, model, args.lr,
                                                                args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if predicted[0].data.cpu().numpy() == args.attack_target:
                    train_success += 1
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1],
                        y_location:y_location + patch.shape[2]]
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        # plt.savefig("E:/Academy/LAB/Diagnosis for Adversaries/Adversarial_Patch_Attack-master/training_pictures/" + str(
        #     epoch) + " patch.png")
        plt.savefig("/home/liyize/Codes/AdvTool/result/Patch/training_results/" + str(
            epoch) + " patch.png")
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch,
                                                                               100 * train_success / train_actual_total))
        train_success_rate = train_success / train_actual_total
        # print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(args.patch_type, args.attack_target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        with open(path1, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_success_rate, test_success_rate])

        if train_success_rate >= best_patch_success_rate and test_success_rate > best_patch_success_test_rate:
            best_patch_success_rate = train_success_rate
            best_patch_success_test_rate = test_success_rate
            best_patch_epoch = epoch
            best_patch = patch
            im = ((np.transpose(patch, (1, 2, 0)) * std + mean) * 255).astype(np.uint8)
            # print(im.shape)
            # im = np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
            im = Image.fromarray(im)
            im.save("/home/liyize/Codes/AdvTool/result/Patch/training_results/best_patch.png")


        # Load the statistics and generate the line
        log_generation(path1)

    print("The best patch is found at epoch {} with success rate {:.3f}% on trainset".format(best_patch_epoch,
                                                                                             100 * best_patch_success_rate))

    advimg_patch(args.patch_type, args.attack_target, best_patch, adv_loader, model, args.model_type, args.task,
                 args.attack_type, args.attack_goal, path0, class_label, args.strategy)
    # print("Patch attack success rate on generating images: {:.3f}%".format(100 * adv_success_rate))
