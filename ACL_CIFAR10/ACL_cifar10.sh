### Pre Training ###
nohup python ACL.py results/natural_full --gpu 0 --dataset cifar10 --method full &

nohup python ACL.py results/adversarial_full --gpu 0 --ACL_DS --dataset cifar10 --method full &

nohup python ACL.py results/RCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method coreset --CoresetLoss KL --fraction 0.05 &
nohup python ACL.py results/RCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method coreset --CoresetLoss KL --fraction 0.1 &
nohup python ACL.py results/RCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method coreset --CoresetLoss KL --fraction 0.2 &

### Finetuning ###
MODEL_PATH = results/adversarial_full/model.pt

nohup python finetune.py cifar100_robust_fully_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset cifar100 --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f1 &
nohup python finetune.py svhn_robust_fully_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset svhn --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f1 &

nohup python finetune.py cifar100_robust_partially_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset cifar100 --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 &
nohup python finetune.py svhn_robust_partially_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset svhn --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 &

nohup python finetune.py cifar100_standard_fully_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset cifar100 --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 --trainmode normal &
nohup python finetune.py svhn_standard_fully_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset svhn --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 --trainmode normal &

nohup python finetune.py cifar100_standard_partially_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset cifar100 --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 --trainmode normal &
nohup python finetune.py svhn_standard_partially_finetune/adversarial_full --gpu 0 --checkpoint $MODEL_PATH --dataset svhn --cvt_state_dict --bnNameCnt 1 --decreasing_lr 40,60 --epochs 100 --fixmode f3 --trainmode normal &

