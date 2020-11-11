import sys
import os

def main():
    if '--area' in sys.argv and '--attack_type' in sys.argv:
        a = sys.argv.index('--attack_type')
        attack_type = sys.argv[a + 1]

        if attack_type == 'PGD_linf' or 'PGD_l2':  # or 'FGSM' or 'CW':
            attack = 'PGD'
        sys.argv[0] = attack + '.py'

        if attack_type == 'Patch':
            attack = 'Patch_Attack'
        sys.argv[0] = attack + '.py'
        
        # More attacks can be added

        command = ' '.join(sys.argv[:])
        command = 'python ' + command
        print(command)
        os.system(command)
        
       
        
    else:
        print('attack type error. Please provide attack type.')


if __name__ == '__main__':
    main()
    
