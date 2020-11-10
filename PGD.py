import torch
import json
import argparse
from imageio import imwrite
from pretrainedmodels import *
from torchvision.models import *
from utils.predict import AttackPredict
from utils.whitebox import WhiteBoxAttack

# from attack.blackbox import BlackBoxAttack


# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--area', type=str, default='C', choices=['C', 'D', 'F'], help='Classification, Detection, '
                                                                                       'Face Recognition')
    parser.add_argument(
        '--attack_type', type=str, default='PGD_linf', choices=['PGD_linf', 'PGD_l2']
    )
    parser.add_argument(
        '--attack_goal', type=str, default='N', choices=['N', 'T'], help='Non-target, Target'
    )
    parser.add_argument("--model_type", type=str, default='resnet18',
                        choices=['inceptionv3', 'resnet18', 'resnet50'],
                        help='pre-trained target classifier')
    parser.add_argument(
        "--strategy", type=str, default='WB', choices=['WB', 'TB', 'QB'], help='White Box, Transfer based black box, '
                                                                               'Query based balck box '
    )
    args = parser.parse_args()

    # Preparation
    print('-' * 75)
    result_str = 'Prediction of {} image:\n[Label:{}]-[Class:{}]-[Confidence:{:.6f}]'

    # Load label-class pairs of ImageNet
    class_label_dict = json.load(open('/data/imagenet_class_index.json'))
    class_label = [class_label_dict[str(k)][1] for k in range(len(class_label_dict))]

    # Source image
    src_image_path = '/data/0.jpg'  # label:762
    # src_image_path = './data/central_perk_299.png'  # label:762
    # print('Source image: [{}]'.format(src_image_path))

    # Model to be attacked
    device = torch.device('cuda')
    if args.model_type == "resnet18":
        model, input_size = resnet18(pretrained=True).to(device), 224
    if args.model_type == "inceptionv3":
        model, input_size = inception_v3(pretrained=True).to(device), 224
    # model, input_size = resnet34(pretrained=True), 224
    # model, input_size = inception_v3(pretrained=True), 299
    print('Model to be attacked: [pretrained {} on ImageNet]'.format(args.model_type))
    print('-' * 75)

    # --------------------------------------------------------------------------------------

    # Prediction of source image
    predictor = AttackPredict(
        model=model, input_size=input_size,
        class_label=class_label, use_cuda=True
    )
    src_label, src_prob, src_class = predictor.run(src_image_path)

    # ======================================================================================

    # White-Box Adversarial Attack on source image
    whitebox_attack = WhiteBoxAttack(
        model=model, input_size=input_size, epsilon=32, alpha=5,
        num_iters=100, early_stopping=5, use_cuda=True
    )

    # 'model' also could be a list of model instances
    # whitebox_attack = WhiteBoxAttack(
    #     model=[resnet18(pretrained=True), resnet34(pretrained=True)],
    #     input_size=input_size, epsilon=16, alpha=5,
    #     num_iters=100, early_stopping=5, use_cuda=True
    # )

    # --------------------------------------------------------------------------------------

    # Non-Targeted Attack
    if args.attack_goal == "N":
        # print('{}'.format(args.attack_type))

        wb_nt_image = whitebox_attack(image_path=src_image_path, label=src_label, norm=args.attack_type, target=False)
        wb_nt_image_path = '/results/{}_{}_{}_{}_{}.jpg'.format(args.area, args.attack_type, args.attack_goal, args.model_type,
                                          args.strategy)
        imwrite(wb_nt_image_path, wb_nt_image)

        wb_nt_label, wb_nt_prob, wb_nt_class = predictor.run(wb_nt_image_path)

        # print(wb_nt_label)
        predictor.run(src_image_path, label2=wb_nt_label)

        src_label, src_prob, src_class, other_label, other_prob, other_class = predictor.run(src_image_path, label2=wb_nt_label)
        print(result_str.format('source', src_label, src_class, src_prob))
        print(result_str.format('non-attack', other_label, other_class, other_prob))
        print('-' * 75)

        print('White-Box Non-Targeted Adversarial Attack')
        wb_nt_label, wb_nt_prob, wb_nt_class, s_label, s_prob, s_class = predictor.run(wb_nt_image_path, src_label)

        print(result_str.format('adversarial', wb_nt_label, wb_nt_class, wb_nt_prob))
        print(result_str.format('sourc', s_label, s_class, s_prob))

   # --------------------------------------------------------------------------------------

    # Targeted Attack
    if args.attack_goal == "T":
        wb_t_image = whitebox_attack(image_path=src_image_path, label=358, norm=args.attack_type, target=True)
        wb_t_image_path = '/results/{}_{}_{}_{}_{}.jpg'.format(args.area, args.attack_type, args.attack_goal, args.model_type, args.strategy)
        imwrite(wb_t_image_path, wb_t_image)

        wb_t_label, wb_t_prob, wb_t_class = predictor.run(wb_t_image_path)

        src_label, src_prob, src_class, other_label, other_prob, other_class = predictor.run(src_image_path, wb_t_label)
        print(result_str.format('source', src_label, src_class, src_prob))
        print(result_str.format('non-attack', other_label, other_class, other_prob))
        print('-' * 75)

        print('White-Box Targeted Adversarial Attack')
        wb_t_label, wb_t_prob, wb_t_class, s_label, s_prob, s_class = predictor.run(wb_t_image_path, src_label)

        print(result_str.format('adversarial', wb_t_label, wb_t_class, wb_t_prob))
        print(result_str.format('sourc', s_label, s_class, s_prob))
        # print('-' * 75)

