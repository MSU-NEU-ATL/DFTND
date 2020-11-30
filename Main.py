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

        # if attack_type == 'trojan':
        #     attack = 'trojan_train'
        # sys.argv[0] = 'DFTND/' + attack + '.py'

        command = ' '.join(sys.argv[:])
        command = 'python ' + command
        print(command)
        os.system(command)
    else:
        print('attack type error. Please provide attack type.')


if __name__ == '__main__':
    main()

