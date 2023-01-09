### Pre-training ###
nohup python SAT.py --gpu 0,1,2,3 --out_dir Natural_full --epsilon 0 --method full &

nohup python SAT.py --gpu 0,1,2,3 --out_dir Random_005 --method random --fraction 0.05 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir Random_01 --method random --fraction 0.1 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir Random_02 --method random --fraction 0.2 &

nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_005 --method coreset --fraction 0.05 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_01 --method coreset --fraction 0.1 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_02 --method coreset --fraction 0.2 &

### finetuning ###
PT=KL_005/checkpoint.pth.tar
NAME=SAT_KL_005

nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_fully_finetuning/${NAME} --dataset cifar10 --lr 0.01 --resume $PT &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_fully_finetuning/${NAME} --dataset cifar100 --lr 0.01 --resume $PT &

nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_partially_finetuning/${NAME} --dataset cifar10 --lr 0.01 --resume $PT --linear &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_partially_finetuning/${NAME} --dataset cifar100 --lr 0.01 --resume $PT --linear &
