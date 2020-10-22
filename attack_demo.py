import numpy as np
import json
import torch
import torch.nn as nn
import torch.utils.data as Data


from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse

# import cw
import torchattacks
import fgsm

def image_folder_custom_label(root, transform, custom_label):
    # custom_label
    # type : List
    # index -> label
    # ex) [ 'goldfish', 'giant_panda', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


# def imshow(img):
#     npimg = img.numpy()
#     # fig = plt.figure(figsize=(5, 15))
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     # plt.title(title)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()


# PGD Attack
def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--area', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                       'Face Recognition')
    parser.add_argument(
        '--attack_name', type=str, default='PGD', choices=['FGSM', 'DeepFool', 'CW', 'PGD']
    )
    parser.add_argument(
        '--attack_goal', type=str, default='N', choices=['N', 'T'], help='Non-target, Target'
    )
    parser.add_argument("--model_type", type=str, default='inceptionv3',
                        choices=['inceptionv3', 'resnet50', 'resnet101', 'vgg16'],
                        help='pre-trained target classifier')
    parser.add_argument(
        "--strategy", type=str, default='WB', choices=['WB', 'TB', 'QB'], help='White Box, Transfer based black box, '
                                                                               'Query based balck box '
    )
    args = parser.parse_args()

    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    Len = len(class_idx)
    # print('Len', Len)
    # print('Class ID', class_idx)
    # print('Label', idx2label)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    ])

    normal_data = image_folder_custom_label(root='./data/imagenet',
                                            transform=transform, custom_label=idx2label)
    normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)

    normal_iter = iter(normal_loader)
    images, labels = normal_iter.next()

    device = torch.device('cuda')
    if args.model_type == "inceptionv3":
        model = models.inception_v3(pretrained=True).to(device)
    elif args.model_type == "vgg16":
        model = models.vgg16(pretrained=True).to(device)
    print("True Image & Predicted Label")

    model.eval()

    correct = 0
    total = 0

    i = 1

for images, labels in normal_loader:
    if args.attack_name == "PGD":
        images = pgd_attack(model, images, labels)
    elif args.attack_name == "CW":
        atk = torchattacks.CW(model, c=1, kappa=0.9, steps=100, lr=0.01)
        images = atk(images, labels)
    elif arg.attack_name == "FGSM"
        images = fgsm.FGSMAttack(model, images, labels, epsilon = 0.3)



    labels = labels.to(device)
    outputs = model(images)

    _, pre = torch.max(outputs.data, 1)

    total += 1
    correct += (pre == labels).sum()

    im = images.cpu().data.squeeze(0)
    # print(im.size())
    im = im.numpy()
    im = (np.transpose(im, (1, 2, 0)) * 255).astype(np.uint8)

    from PIL import Image

    im = Image.fromarray(im)


    im.save('./Label_demo1/{}_{}_{}_{}_{}_{}.jpg'.format(args.area,
                                                                  args.attack_name,
                                                                  args.attack_goal,
                                                                  args.model_type,
                                                                  args.strategy, i))
    i = i + 1
