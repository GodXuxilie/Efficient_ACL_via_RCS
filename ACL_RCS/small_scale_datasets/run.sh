### Efficient Pre-Training via RCS on CIFAR10 ###
nohup python ACL_RCS.py ACL_RCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py ACL_RCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py ACL_RCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py DynACL_RCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py DynACL_RCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py DynACL_RCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

### Efficient Pre-Training via RCS on CIFAR100 ###
nohup python ACL_RCS.py ACL_RCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py ACL_RCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py ACL_RCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py DynACL_RCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py DynACL_RCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py DynACL_RCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

### Efficient Pre-Training via RCS on STL10 ###
nohup python ACL_RCS.py ACL_RCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL_RCS.py ACL_RCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL_RCS.py ACL_RCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL_RCS.py DynACL_RCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL_RCS.py DynACL_RCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL_RCS.py DynACL_RCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &


### Finetuning ###
EXP_NAME=./ACL_RCS_cifar10_r18_cifar10
MODEL_PATH=./ACL_RCS_KL_005/model.pt
DOWNSTREAM_TASK=cifar10
PRETRAING_METHOD=DynACL_RCS

python finetuning.py --gpu 2 --experiment EXP_NAME --dataset DOWNSTREAM_TASK --pretraining PRETRAING_METHOD --model r18 --checkpoint MODEL_PATH --mode ALL --eval-AA --eval-OOD


