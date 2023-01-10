### Pre-training ###
nohup python ACL.py results/Standard_full --gpu 0 --method Entire &

nohup python ACL.py results/ACL_random_005 --ACL_DS --gpu 0 --method Random --fraction 0.05 &

nohup python ACL.py results/ACL_KL_005 --ACL_DS --gpu 0 --method RCS --fraction 0.05 &

### Finetuning ###
PT=results/ACL_KL_005/model.pt
NAME=ACL_KL_005

nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_fully_finetuning/${NAME} --dataset cifar10 --lr 0.1 --resume $PT &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_fully_finetuning/${NAME} --dataset cifar100 --lr 0.1 --resume $PT &
nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_partilly_finetuning/${NAME} --linear --dataset cifar10 --lr 0.01 --resume $PT &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_partilly_finetuning/${NAME} --linear --dataset cifar100 --lr 0.01 --resume $PT &

nohup python transfer.py --gpu 0 --out_dir cifar10_standard_fully_finetuning/${NAME} --dataset cifar10 --lr 0.1 --resume $PT &
nohup python transfer.py --gpu 0 --out_dir cifar100_standard_fully_finetuning/${NAME} --dataset cifar100 --lr 0.1 --resume $PT &
nohup python transfer.py --gpu 0 --out_dir cifar10_standard_partilly_finetuning/${NAME} --linear --dataset cifar10 --lr 0.01 --resume $PT &
nohup python transfer.py --gpu 0 --out_dir cifar100_standard_partilly_finetuning/${NAME} --linear --dataset cifar100 --lr 0.01 --resume $PT &

