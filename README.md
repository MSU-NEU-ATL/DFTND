# attack demo


To generate attacked images, use the following example. Make sure to specify the attack_type and other commands if necessary. 
```bash
python main.py --area C --attack_type PGD_l2 --attack_goal N --model_type resnet18 --strategy WB 
```

To submit your attacks, name your main attack file with the attack name such as 'PGD.py'. Originize your source files and make sure the attack can run such as the following example.
```bash
python PGD.py
```
