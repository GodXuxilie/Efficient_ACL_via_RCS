import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import sys,os
path = os.path.dirname(os.path.dirname(__file__)) 
print(path)
sys.path.append(path)
# from models.resnet_multi_bn import resnet18, proj_head
from utils import *
import torchvision.transforms as transforms
import os
import numpy as np
from optimizer.lars import LARS
import datetime
from coreset_util import LossCoreset, RandomCoreset
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str, help='location for saving trained models')
parser.add_argument('--data', type=str, default='/home/x/xuxilie/data', help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be used, (cifar10 or cifar100)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.6, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--ACL_DS', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')
parser.add_argument('--twoLayerProj', action='store_true', help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--epsilon', default=8, type=float, help='number of total epochs to run')

parser.add_argument('--method', default='coreset', type=str, choices=['random', 'coreset', 'full'])
parser.add_argument('--fre', type=int, default=10, help='')
parser.add_argument('--warmup', type=int, default=20, help='')
parser.add_argument('--fraction', type=float, default=0.05, help='')
parser.add_argument('--CoresetLoss', type=str, default='KL', help='if specified, use pgd dual mode,(cal both adversarial and clean)', choices=['KL', 'JS', 'ot'])
parser.add_argument('--Coreset_num_steps',  default=1, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--Coreset_lr',  default=0.01, type=float, help='how many iterations employed to attack the model')
parser.add_argument('--val_frac', type=float, default=0.01, help='')

class proj_head_module(nn.Module):
    def __init__(self, ch, twoLayerProj=False):
        super(proj_head_module, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = torch.nn.BatchNorm1d(ch)
        self.fc2 = nn.Linear(ch, ch, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # debug
        # print("adv attack: {}".format(flag_adv))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr


def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    save_dir = os.path.join('checkpoints_valtest', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    num_classes = 1000
    print('==> Load Model')
    from models.wrn import WideResNet
    model = WideResNet(28, num_classes, 10, dropRate=0).cuda()

    h_size = model.nChannels
    proj_head = proj_head_module(h_size).cuda()
    model.linear = proj_head
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    strength = 1.0
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
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

    from PIL import Image
    from torch.utils.data import Dataset
    from typing import TypeVar, Sequence

    from imagenet_downsampled import ImageNetDS
    class CustomImageNetDS(ImageNetDS):
        def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
            super().__init__(**kwds)
            self.withLabel = withLabel
            self.labelTrans = labelTrans

            if labelSubSet is not None:
                self.data = self.data[labelSubSet]

        def __getitem__(self, index):
            # to return a PIL Image
            img = self.train_data[index]
            img = Image.fromarray(img)
            imgs = [self.transform(img), self.transform(img)]
            if not self.withLabel:
                # return self.transform(img), self.transform(img)
                return torch.stack(imgs)
            else:
                imgLabelTrans = self.labelTrans(img)
                label = self.targets[index]
                return torch.stack(imgs), imgLabelTrans, label

    trainset = CustomImageNetDS(root='./data/imagenet32', img_size=32, train=True, transform=tfs_train)
    num_classes = 1000
        
    full_indices = np.arange(0,len(trainset),1)
    train_indices = np.random.choice(len(trainset), size=int(len(trainset) * 0.995), replace=False)
    val_indices = np.delete(full_indices, train_indices)
    validset = Subset(trainset, val_indices)
    trainset = Subset(trainset, train_indices)
    print(len(trainset), len(validset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=2 * len(train_loader))
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            # optimizer.load_state_dict(checkpoint['optim'])
            optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    if args.method == 'random':
        coreset_class = RandomCoreset(trainset, fraction=args.fraction, log=log, args=args, model=model)
    elif args.method == 'coreset':
        coreset_class = LossCoreset(trainset, fraction=args.fraction, validation_loader=validation_loader, model=model, args=args, log=log)
    
    valid_loss_list = []
    test_loss_list = []
    for epoch in range(start_epoch, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        starttime = datetime.datetime.now()
        if args.method != 'full' and epoch >= args.warmup and (epoch-1) % args.fre == 0:
            tmp_state_dict = model.state_dict()
            if args.adaptive_lr:
                coreset_class.lr = optimizer.state_dict()['param_groups'][0]['lr']
            else:
                coreset_class.lr = args.Coreset_lr
            coreset_class.model.load_state_dict(tmp_state_dict)
            train_loader = coreset_class.get_subset_loader()
            model.load_state_dict(tmp_state_dict)
            for name, param in model.named_parameters():
                param.requires_grad = True 
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(step,
                                                        args.epochs * len(train_loader),
                                                        1,  # since lr_lambda computes multiplicative factor
                                                        1e-6 / args.lr,
                                                        warmup_steps=2 * len(train_loader))
            )
            for i in range((epoch - 1) * len(train_loader)):
                scheduler.step()
        elif args.method != 'full' and epoch < args.warmup:
            train_loader = coreset_class.load_subset_loader()
        elif args.method != 'full':
            log.info('train on the sub set')
        if args.scheduler == 'cosine' and epoch >= 2:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(step,
                                                        args.epochs * len(train_loader),
                                                        1,  # since lr_lambda computes multiplicative factor
                                                        1e-6 / args.lr,
                                                        warmup_steps=2 * len(train_loader))
            )
            for i in range((epoch - 1) * len(train_loader)):
                scheduler.step()
        log.info('train on the full set')

        train_loss = train(train_loader, model, optimizer, scheduler, epoch, log, num_classes=num_classes)
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'valid_loss_list': valid_loss_list,
            'test_loss_list': test_loss_list,
        }, filename=os.path.join(save_dir, 'model.pt'))

        log.info('[Epoch: {}] [Train loss: {}] [Train time: {}]'.format(epoch, train_loss, time))

def train(train_loader, model, optimizer, scheduler, epoch, log, num_classes):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
        print(inputs.shape)
        if not args.ACL_DS:
            features = model.train()(inputs)
            loss = nt_xent(features, t=0.1)
        else:
            inputs_adv = PGD_contrastive(model, inputs, iters=args.pgd_iter, singleImg=False,eps=args.epsilon/255,alpha=(args.epsilon/4)/255)
            features_adv = model.train()(inputs_adv)
            features = model.train()(inputs)
            loss = (nt_xent(features, t=0.1) + nt_xent(features_adv, t=0.1))/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # break

        # torch.cuda.empty_cache()
        print_freq = max(int(len(train_loader) / 5), 1)
        print(print_freq)
        if i % print_freq == 0:
        # if i % 1 == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

    return losses.avg


def validate(train_loader, val_loader, model, log, num_classes=10):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.fc
    ch = model.fc.in_features
    model.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 100
    lr = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                epochs_max * len(train_loader),
                                                1,  # since lr_lambda computes multiplicative factor
                                                1e-6 / lr,
                                                warmup_steps=0)
    )

    for epoch in range(epochs_max):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (sample) in enumerate(train_loader):
            scheduler.step()

            x, y = sample[0].cuda(), sample[1].cuda()
            p = model.eval()(x)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(float(loss.detach().cpu()))

            train_time = time.time() - end
            end = time.time()
            train_time_meter.update(train_time)

        log.info('Test epoch: ({0})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'train_time: {train_time.avg:.2f}\t'.format(
                    epoch, loss=losses, train_time=train_time_meter))

    acc = []
    for loader in [train_loader, val_loader]:
        losses = AverageMeter()
        losses.reset()
        top1 = AverageMeter()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs)
                loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if i % args.print_freq == 0:
                log.info('Test: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                             i, len(loader), loss=losses, top1=top1))

        acc.append(top1.avg)

    # recover every thing
    model.fc = previous_fc
    model.cuda()
    for param in model.parameters():
        param.requires_grad = True

    return acc


def cal_train_loss(train_loader, model, optimizer, scheduler, epoch, log, num_classes):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)
        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()

        if not args.ACL_DS:
            with torch.no_grad():
                features = model.train()(inputs)
                loss = nt_xent(features, t=0.1)
        else:
            inputs_adv = PGD_contrastive(model, inputs, iters=args.pgd_iter, singleImg=False)
            with torch.no_grad():
                features_adv = model.train()(inputs_adv)
                features = model.train()(inputs)
                loss = (nt_xent(features, t=0.1) + nt_xent(features_adv, t=0.1))/2

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

    return losses.avg

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def PGD_contrastive(model, inputs, eps=4. / 255., alpha=1. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)
    model.eval()
    if singleImg:
        # project half of the delta to be zero
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    for i in range(iters):
        if feature_gene is None:
            if sameBN:
                features = model(inputs + delta)
            else:
                features = model(inputs + delta)
        else:
            features = feature_gene(model, inputs + delta)

        model.zero_grad()
        loss = nt_xent(features, t=0.1)
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

        if singleImg:
            # project half of the delta to be zero
            idx = [i for i in range(1, delta.data.shape[0], 2)]
            delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    return (inputs + delta).detach()


if __name__ == '__main__':
    main()


