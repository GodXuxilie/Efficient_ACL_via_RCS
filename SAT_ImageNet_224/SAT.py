import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
import numpy as np
import logging
import torch
from utils_log.utils import set_logger
import attack_generator as attack
from coreset_util import RCS, RandomSelection

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=4, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=3, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=2/255, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="ResNet18",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--dataset', type=str, default="imagenet", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./results/AT', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--nes', type=bool, default=False)
parser.add_argument('--ams', type=bool, default=False)

parser.add_argument('--method', default='random', type=str, choices=['Random', 'RCS', 'Entire'])
parser.add_argument('--fre', type=int, default=10, help='')
parser.add_argument('--warmup', type=int, default=10, help='')
parser.add_argument('--fraction', type=float, default=0.1, help='')
parser.add_argument('--CoresetLoss', type=str, default='KL', help='if specified, use pgd dual mode,(cal both adversarial and clean)', choices=['KL', 'JS', 'ot'])
parser.add_argument('--Coreset_lr',  default=0.01, type=float, help='how many iterations employed to attack the model')
parser.add_argument('--Coreset_epsilon', type=float, default=4, help='perturbation bound')
parser.add_argument('--Coreset_num_steps', type=int, default=1, help='maximum perturbation step K')
parser.add_argument('--Coreset_step_size', type=float, default=2/255, help='step size')

args = parser.parse_args()

args.epsilon = args.epsilon / 255
args.Coreset_epsilon = args.Coreset_epsilon / 255

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

out_dir = args.out_dir + '_{}_{}'.format(args.net,args.dataset)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

set_logger(os.path.join(out_dir, 'training.log'))

logging.info(out_dir)
logging.info(args)

def train(model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        print(data.shape)
        # Get Most adversarial training data via PGD
        if args.epsilon != 0:
            x_adv = attack.pgd(model,data,target,epsilon=args.epsilon,step_size=args.epsilon * 2 / args.num_steps,num_steps=args.num_steps,
                            loss_fn='cent',category='Madry',rand_init=True)
        else:
            x_adv = data

        model.train()
        optimizer.zero_grad()
        output = model(x_adv)

        # calculate standard adversarial training loss
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, target)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum

# setup data loader
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

from torch.utils.data import Dataset
from typing import TypeVar, Sequence
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

print('==> Load Data')
trainset= torchvision.datasets.ImageNet('../datasets', split='train', transform=transform_train)
testset = torchvision.datasets.ImageNet('../datasets', split='val', transform=transform_test)
num_classes = 1000
    
full_indices = np.arange(0,len(trainset),1)
train_indices = np.random.choice(len(trainset), size=int(len(trainset) * 0.995), replace=False)
val_indices = np.delete(full_indices, train_indices)
validset = Subset(trainset, val_indices)
trainset = Subset(trainset, train_indices)
print(len(trainset), len(validset))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


print('==> Load Model')
from torchvision.models.resnet import ResNet, Bottleneck
class ResNet_new(ResNet):
    def __init__(
    self,
    block,
    layers,
    num_classes = 1000,
    zero_init_residual = False,
    groups = 1,
    width_per_group = 64,
    replace_stride_with_dilation = None,
    norm_layer = None,
    ) -> None:
        super().__init__(block,
    layers,
    num_classes,
    zero_init_residual,
    groups,
    width_per_group,
    replace_stride_with_dilation,
    norm_layer)

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

print('==> Load Model')
model = ResNet_new(Bottleneck, [3, 4, 6, 3]).cuda()
logging.info(args.net)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)

def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 60:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr 

if len(args.gpu.split(',')) > 1:
    model = torch.nn.DataParallel(model)

start_epoch = 0
# Resume
title = 'AT train'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    logging.info('==> Adversarial Training Resuming from checkpoint ..')
    logging.info(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(start_epoch)

test_nat_acc = 0
test_pgd10_acc = 0
best_epoch = 0

if args.method == 'Random':
    coreset_class = RandomSelection(trainset, fraction=args.fraction, log=logging, args=args, model=model)
elif args.method == 'RCS':
    coreset_class = RCS(trainset, fraction=args.fraction, validation_loader=valid_loader, model=model, args=args, log=logging)
    
test_natloss_list = []
test_natacc_list = []

for epoch in range(start_epoch, args.epochs):
    lr = adjust_learning_rate(optimizer, epoch + 1)
    if args.method != 'Entire' and epoch >= args.warmup and (epoch - start_epoch) % args.fre == 0:
        tmp_state_dict = model.state_dict()
        coreset_class.lr = args.Coreset_lr
        coreset_class.model.load_state_dict(tmp_state_dict)
        train_loader = coreset_class.get_subset_loader()
        for params in model.parameters():
            params.requires_grad = True
        model.load_state_dict(tmp_state_dict)
    elif args.method != 'Entire' and epoch > args.warmup:
        train_loader = coreset_class.load_subset_loader()
        logging.info('train on the previously subset')
    else:
        logging.info('train on the entire set')

    train_time, train_loss = train(model, train_loader, optimizer)

    model.eval()
    loss, test_nat_acc = attack.eval_clean(model, test_loader)
    test_natloss_list.append(loss)
    test_natacc_list.append(test_nat_acc)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_pgd10_acc': test_pgd10_acc,
        'test_natloss_list': test_natloss_list,
        'test_natacc_list': test_natacc_list,
    })

    logging.info(
        'Epoch: [%d | %d] | Train Time: %.2f s | Train Loss %.4f | Natural Test Acc %.4f\n' % (
        epoch + 1,
        args.epochs,
        train_time,
        train_loss,
        test_nat_acc,
        )
    )

    
