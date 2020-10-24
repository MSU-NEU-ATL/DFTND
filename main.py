import sys
import os

def main():

    if '--attack_type' in sys.argv:
        a = sys.argv.index('--attack_type')
        attack_type = sys.argv[a+1]
        if attack_type == 'PGD' or 'FGSM' or 'CW':
            attack_type = 'PGD'
        if attack_type == 'sturctured' or 'ADMM':
            attack_type = 'sturctured'
        sys.argv[0] = attack_type + '.py'

        command = ' '.join(sys.argv[:])
        command = 'python ' + command
        print(command)
        os.system(command)
    else:
        print('attack type error. Please provide attack type.')
    # if len(sys.argv) > 1:
    #     command = ' '.join(sys.argv[:])
    # else:
    #     command = str(sys.argv[0])
    # print(command)

if __name__ == '__main__':
    main()
