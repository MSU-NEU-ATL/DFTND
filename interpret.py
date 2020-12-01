from keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from utils.utils_interpret import IntegratedGradientsAttack
from utils.modified_squeezenet import SqueezeNet
from utils.utils_interpret import dataReader
import os

parser = argparse.ArgumentParser(description='Adversarial Attack',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--task', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                   'Face Recognition')
parser.add_argument(
    '--attack_type', type=str, default='PGD_linf', choices=['FGSM', 'CW', 'PGD', 'Patch', 'trojan', 'interpret']
)
parser.add_argument(
    '--attack_goal', type=str, default='N', choices=['N', 'T'], help='Non-target, Target'
)
parser.add_argument("--model_type", type=str, default='SqueezeNet',
                    choices=['SqueezeNet'],
                    help='pre-trained target classifier')
parser.add_argument("--method", type=str, default='mass_center',
                    choices=['mass_center', 'random', 'topK'],
                    help='pre-trained target classifier')
parser.add_argument(
    "--strategy", type=str, default='WB', choices=['WB', 'TB', 'QB'], help='White Box, Transfer based black box, '
                                                                           'Query based balck box ')
parser.add_argument('--epsilon', default=8, type=int, metavar='N',
                    help='epsilon perturbation constraint')

parser.add_argument('--adv_size', type=int, default=2, help="number of generating adversarial images")

parser.add_argument('--result_dir', type=str, default='result/interpret', help="dir of the result")

parser.add_argument('--test', action='store_true', default=False,
                    help='check the result (True) or save the result (False)')
args = parser.parse_args()

def get_session(number=None):
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    return tf.Session(config=config_gpu)


if not os.path.isdir("result"):
    os.mkdir("result")
if not os.path.isdir(args.result_dir):
    os.mkdir(args.result_dir)

X_dic, y_dic, labels_dic = dataReader()
mean_image = np.zeros((227,227,3))
mean_image[:,:,0]=103.939
mean_image[:,:,1]=116.779
mean_image[:,:,2]=123.68
X = X_dic - mean_image #Mean Subtraction
y = y_dic

tf.reset_default_graph()
sess = get_session()
K.set_session(sess)
K.set_learning_phase(0)

if args.model_type == 'SqueezeNet':
    model2 = SqueezeNet("softplus")
    model = SqueezeNet("relu") ## Surrogate model (Refer to the original paper)
else:
    print('model not supported')

def create_saliency_ops(NET):

    w = NET.input.get_shape()[1].value
    h = NET.input.get_shape()[2].value
    c = NET.input.get_shape()[3].value
    num_classes = NET.output.get_shape()[-1].value
    NET.label_ph = tf.placeholder(tf.int32,shape=())
    NET.reference_image = tf.placeholder(tf.float32,shape=(w,h,c))
    sum_logits = tf.reduce_sum(NET.output*tf.expand_dims(tf.one_hot(NET.label_ph,num_classes),0))
    parallel_gradients = tf.gradients(sum_logits,NET.input)[0]
    average_gradients = tf.reduce_mean(parallel_gradients,0)
    difference_multiplied = average_gradients * (NET.input[-1]-NET.reference_image)
    saliency_unnormalized = tf.reduce_sum(tf.abs(difference_multiplied),-1)
    NET.saliency = w*h*tf.divide(saliency_unnormalized,tf.reduce_sum(saliency_unnormalized))
    # we multiply the normalized salinecy map with image size to make saliency scores of
    #images of different sizes comparable
    NET.saliency_flatten = tf.reshape(NET.saliency,[w*h])

reference_image = np.zeros((227,227,3)) #Our chosen reference(the mean image)
create_saliency_ops(model)
create_saliency_ops(model2)

for i in range(args.adv_size):
    if i > X.shape[0]:
        break
    # n = np.random.choice(X.shape[0])
    n=i
    original_label = y[n]
    test_image = X[n]
    plt.rcParams["figure.figsize"]=8,8
    print("Image Label : {}".format(labels_dic[y[n]]))
    plt.imshow((X[n,:,:,::-1]+mean_image[:,:,::-1])/255)
    if args.test:
        plt.show()


    k_top = 1000 #Recommended for ImageNet
    num_steps = 100#Number of steps in Integrated Gradients Algorithm (refer to the original paper)
    module = IntegratedGradientsAttack(sess, mean_image, test_image, original_label,NET=model, NET2=model2, k_top=k_top,
                                     num_steps=num_steps,reference_image=reference_image)

    method = args.method  #Method should be one of "random", "mass_center", "topK"
    # method = "mass_center" #Method should be one of "random", "mass_center", "topK"

    epsilon = args.epsilon #Maximum allowed perturbation for each pixel
    # epsilon = 8 #Maximum allowed perturbation for each pixel

    output = module.iterative_attack(method, epsilon=epsilon, alpha=0.5, iters=5, measure="mass_center")
    print("The prediction confidence changes from {} to {} after perturbation.".format(module.original_confidence,output[-1]))
    print('''{} % of the {} most salient pixels in the original image are among {} most salient pixels of the 
    perturbed image'''.format(output[0]*100,k_top,k_top))
    print("The rank correlation between salieny maps is equal to {}".format(output[1]))
    print("The L2 distance between mass centers of saliencies is {} pixels.".format(output[2]))

    gradient = np.mean(sess.run(tf.gradients(tf.reduce_sum(model.layers[-1].input[:,original_label]),model.input)[0],
                        {model.input:[float(i+1)/num_steps * (test_image-reference_image) + reference_image\
                                     for i in range(num_steps)]}),0)

    if args.test:
        mpl.rcParams["figure.figsize"]=8,8
        plt.rc("text",usetex=False)
        plt.rc("font",family="sans-serif",size=12)
        saliency = np.sum(np.abs(gradient*(test_image-reference_image)),-1)
        original_saliency = 227*227*saliency/np.sum(saliency)
        plt.subplot(2,2,1)
        plt.title("Original Image")
        image = X[n,:,:,::-1]+mean_image[:,:,::-1]
        plt.imshow(image/255)
        plt.subplot(2,2,2)
        plt.title("Original Image Saliency Map")
        plt.imshow(original_saliency,cmap="hot")
        gradient = np.mean(sess.run(tf.gradients(tf.reduce_sum(model.layers[-1].input[:,original_label]),model.input)[0],
                            {model.input:[float(i+1)/num_steps * (module.perturbed_image-reference_image) + reference_image\
                                         for i in range(num_steps)]}),0)
        saliency = np.sum(np.abs(gradient*(module.perturbed_image-reference_image)),-1)
        perturbed_saliency = 227*227*saliency/np.sum(saliency)
        plt.subplot(2,2,3)
        plt.title("Perturbed Image")
        perturbed_image = (module.perturbed_image[:,:,::-1]+mean_image[:,:,::-1])
        # plt.savefig('testplot.png')
        plt.imshow(perturbed_image/255)
        plt.subplot(2,2,4)
        plt.title("Perturbed Image Saliency Map")
        plt.imshow(perturbed_saliency,cmap="hot")
        plt.show()
    else:
        perturbed_image = (module.perturbed_image[:, :, ::-1] + mean_image[:, :, ::-1])
        plt.axis('off')
        plt.imshow((X[n, :, :, ::-1] + mean_image[:, :, ::-1]) / 255)
        plt.savefig('./result/interpret/{}_{}_{}_{}_{}_{}.png'.format(args.task, args.attack_type, args.attack_goal,
                                                                      args.model_type, args.strategy, i))
