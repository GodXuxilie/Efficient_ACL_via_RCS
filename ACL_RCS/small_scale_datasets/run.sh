### Pre Training on CIFAR10 ###
nohup python ACL_RCS.py results/ACL_RCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py results/ACL_RCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py results/ACL_RCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py results/DynRCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py results/DynRCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py results/DynRCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

### Pre Training on CIFAR100 ###
nohup python ACL_RCS.py results/ACL_RCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py results/ACL_RCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py results/ACL_RCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py results/DynRCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py results/DynRCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py results/DynRCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

### Pre Training on STL10 ###
nohup python ACL_RCS.py results/ACL_RCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py results/ACL_RCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py results/ACL_RCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py results/DynRCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py results/DynRCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py results/DynRCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &


### Finetuning ###
cd finetune_eval
MODEL_PATH=../results/ACL_RCS_KL_005/model.pt
DOWNSTREAM_TASK=cifar10
SOURCE_DATASET=cifar10

### SLF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint $MODEL_PATH --dataset $DOWNSTREAM_TASK --source $SOURCE_DATASET --cvt_state_dict --bnNameCnt 1 --evaluation_mode SLF
### ALF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint $MODEL_PATH --dataset $DOWNSTREAM_TASK --source $SOURCE_DATASET --cvt_state_dict --bnNameCnt 1 --evaluation_mode ALF
### AFF ###
python test_AFF.py --experiment exp_name --gpu 0 --checkpoint $MODEL_PATH --dataset $DOWNSTREAM_TASK --source $SOURCE_DATASET


