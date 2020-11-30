import sys
import os


def main():
    if '--task' in sys.argv and '--attack_type' in sys.argv:
        a = sys.argv.index('--attack_type')
        attack_type = sys.argv[a + 1]

        if attack_type == 'PGD':
            attack = 'PGD_Attack'

        if attack_type == 'Patch':
            attack = 'Patch_Attack'

        if attack_type == 'CW':
            attack = 'CW_Attack'

        if attack_type == 'FGSM':
            attack = 'FGSM_Attack'

        sys.argv[0] = attack + '.py'

        if attack_type == 'trojan':
            attack = 'trojan_train'
        sys.argv[0] = 'DFTND/' + attack + '.py'

        command = ' '.join(sys.argv[:])
        command = 'python ' + command
        print(command)
        os.system(command)
        
    elif '--attack_type' in sys.argv:
        a = sys.argv.index('--attack_type')
        attack_type = sys.argv[a + 1]
        
        if attack_type =='SNGAN':
            command = ' '.join(sys.argv[3:])
            command = 'python ' + command
            print(command)
            os.system('ls')
            os.chdir('./SNGAN/')
            os.system('pwd')
        
        if attack_type =='FSGAN':
            command = ' '.join(sys.argv[3:])
            print(command)
            os.system('ls')
            os.chdir('./FSGAN/')
            os.system('pwd')
            
        if attack_type =='SEAN':
            #command = 'python ' + './evaluations/gen_images.py --config_path ./configs/sn_projection_celeba.yml --snapshot ./models/celeba_align_png_cropped.npz --results_dir ./gen_image/ --num_pngs 1000 --seed 0'
            
            command = ' '.join(sys.argv[3:])
            print(command)
            os.system('ls')
            os.chdir('./SEAN/')
            os.system('pwd')
            
        if attack_type =='PGAN':
            command = ' '.join(sys.argv[3:])
            command = 'python ' + command
            print(command)
            os.system('ls')
            os.chdir('./PGAN/')
            os.system('pwd')
        
        
        
        os.system(command)
       
        
    else:
        print('attack type error. Please provide attack type.')


if __name__ == '__main__':
    main()
