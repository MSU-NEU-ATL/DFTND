import os
import sys
import torch
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
from robustness.tools.constants import CLASS_DICT
from user_constants import DATA_PATH_DICT
import math
import torchvision.transforms as transforms
import torchvision
import dataset_input
import utilities
import argparse
import json
from tqdm import trange

DATASETS = {
    'imagenet': 'ImageNet',
    'restricted_imagenet': 'RestrictedImageNet',
    'restricted_imagenet_balanced': 'RestrictedImageNetBalanced',
    'cifar': 'CIFAR',
    'cifar10': 'CIFAR',
    'cinic': 'CINIC',
    'a2b': 'A2B',
}

def updateJsonFile(file, method, dataset, path):
    jsonFile = open(file, "r") # Open the JSON file for reading
    config = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file
    
    data = config['data']
    data["poison_method"] = method
    data["dataset"] = dataset
    data["path"] = path
    
    ## Save our changes to JSON file
    jsonFile = open(file, "w+")
    jsonFile.write(json.dumps(config))
    jsonFile.close()

# set the parser
parser = argparse.ArgumentParser(description='Trojan Attack')
parser.add_argument('--area', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                       'Face Recognition')
parser.add_argument(
        '--attack_type', type=str)
parser.add_argument('--model', type=str, default='resnet50', help='model going to use')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['imagenet', 'restricted_imagenet', 'restricted_imagenet_balanced', 'cifar', 'cinic', 'a2b', 'cifar10'], 
                    help='dataset going to be used')
parser.add_argument('poison_method', type=str, default='pattern', choices=['pattern', 'pattern2', 'pixel', 'square', 'ell'], 
                    help='trigger pattern')
parser.add_argument('path', type=str, default='cifar10', help='path to the dataset')
args = parser.parse_args()

updateJsonFile('config_traincifar.json', args.poison_method, args.dataset, args.path)
# print out the poison method and the trigger pattern
config = utilities.config_to_namedtuple(utilities.get_config('config_traincifar.json'))
print("The poison method is " + str(config.data.poison_method))
train_images = np.zeros((32, 32, 3), dtype='uint8')
position = config.data.position
color = config.data.color
method = config.data.poison_method
poison_exp = dataset_input.poison(train_images, method, position, color)
print("The trigger pattern is ")
plt.imshow(poison_exp, interpolation='nearest')
plt.imsave("trigger_pattern.png", poison_exp)



model_dir = config.model.output_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
device = torch.device('cuda')

# Setting up training parameters
max_num_training_steps = config.training.max_num_training_steps
step_size_schedule = config.training.step_size_schedule
weight_decay = config.training.weight_decay
momentum = config.training.momentum
batch_size = 64
eval_during_training = config.training.eval_during_training
num_clean_examples = config.training.num_examples
if eval_during_training:
    num_eval_steps = config.training.num_eval_steps

# Setting up output parameters
num_output_steps = config.training.num_output_steps
num_summary_steps = config.training.num_summary_steps
num_checkpoint_steps = config.training.num_checkpoint_steps

from dataset_wrapper import wrapper

dataset = wrapper()

#
# Load model
model_kwargs = {
    'arch': args.model,
    'dataset': getattr(datasets, DATASETS[args.dataset])(args.path),
}

# model_kwargs = {
#     'arch': 'resnet50',
#     'dataset': datasets.RestrictedImageNet('imagenet'),

# }
model_kwargs['state_dict_path'] = 'model'
model, _ = model_utils.make_and_restore_model(**model_kwargs)
# model.eval()
# pass



# model = dataset.get_model(arch='resnet50')
model = model.to(device=device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)



correct = 0
total = 0
train_loss = 0
best = 0

for ii in range(max_num_training_steps + 1):
    model.train()
    x_batch, y_batch = dataset.train_data.get_next_batch(batch_size,
                                                         multiple_passes=True)
    x_batch = x_batch / 255.0
    inputs = torch.from_numpy(x_batch.astype(np.float32).transpose((0, 3, 1, 2))).cuda()
    targets = torch.from_numpy(y_batch.astype(np.int64)).cuda()
    optimizer.zero_grad()
    outputs, _ = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    if ii % num_output_steps == 0:
        print(f'step: {ii}')
        print(f'Train loss: {train_loss / (ii + 1)}')
        print(f'Accuracy: {correct/total}')



    if eval_during_training and ii % num_eval_steps == 0:
        model.eval()

        print(f'------evaluating----- step: {ii}')
        eval_batch_size = config.eval.batch_size
        poison_method = config.data.poison_method
        clean_label = config.data.clean_label
        target_label = config.data.target_label
        position = config.data.position
        color = config.data.color
        print("poison_method " + str(poison_method) + "\n")
        print("clean label " + str(clean_label) + "\n")
        print("target label " + str(target_label) + "\n")
        print("position " + str(position) + "\n")
        print("color " + str(color) + "\n")
        #print(poison_method, clean_label, target_label, position, color)
        num_eval_examples = len(dataset.eval_data.xs)
        num_clean_examples = 0
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        total_xent_nat = 0.
        total_corr_nat = 0
        total_xent_pois = 0.
        total_corr_pois = 0

        for ibatch in trange(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch_eval = dataset.eval_data.xs[bstart:bend, :] / 255.0
            y_batch_eval = dataset.eval_data.ys[bstart:bend]
            pois_x_batch_eval = dataset.poisoned_eval_data.xs[bstart:bend, :] / 255.0
            pois_y_batch_eval = dataset.poisoned_eval_data.ys[bstart:bend]

            inputs = torch.from_numpy(x_batch_eval.astype(np.float32).transpose((0, 3, 1, 2))).cuda()
            targets = torch.from_numpy(y_batch_eval.astype(np.int64)).cuda()

            with torch.no_grad():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)

            total_xent_nat += loss.item()
            total_corr_nat += predicted.eq(targets).sum().item()

            if clean_label > -1:
                clean_indices = np.where(y_batch_eval == clean_label)[0]
                if len(clean_indices) == 0: continue
                pois_x_batch_eval = pois_x_batch_eval[clean_indices]
                pois_y_batch_eval = np.repeat(target_label, len(clean_indices))
            else:
                pois_y_batch_eval = np.repeat(target_label, bend - bstart)
            num_clean_examples += len(pois_x_batch_eval)

            inputs = torch.from_numpy(pois_x_batch_eval.astype(np.float32).transpose((0, 3, 1, 2))).cuda()
            targets = torch.from_numpy(pois_y_batch_eval.astype(np.int64)).cuda()

            with torch.no_grad():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)

            total_xent_pois += loss.item()
            total_corr_pois += predicted.eq(targets).sum().item()

        avg_xent_nat = total_xent_nat / num_eval_examples
        acc_nat = total_corr_nat / num_eval_examples
        avg_xent_pois = total_xent_pois / num_clean_examples
        acc_pois = total_corr_pois / num_clean_examples

        print('Eval at step: {}'.format(ii))
        print('  natural: {:.2f}%'.format(100 * acc_nat))
        print('  avg nat xent: {:.4f}'.format(avg_xent_nat))
        print('  poisoned: {:.2f}%'.format(100 * acc_pois))
        print('  avg pois xent: {:.4f}'.format(avg_xent_pois))

        # Write a checkpoint
        if acc_nat > best:
            best = acc_nat
            CKPTS_SCHEMA = {
                'epoch': ii,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(CKPTS_SCHEMA, 'models/cifarpert.pt')
            print('saved')

pass