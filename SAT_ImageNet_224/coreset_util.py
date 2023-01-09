from operator import index
from pickletools import optimize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import TypeVar, Sequence
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

import datetime
import copy
import math
import torch.nn.functional as F

import attack_generator as attack
# torch.autograd.set_detect_anomaly(True)

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


class IndexSubset(Dataset[T_co]):
    r"""
    Subset of a dataset with weights at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        weights (sequence): Weights of the subset
    """
    dataset: Dataset[T_co]

    def __init__(self, dataset: Dataset[T_co]) -> None:
        self.dataset = dataset

    def __getitem__(self, idx):
        # except:
        tmp_list = []
        tmp_list.append(idx)
        tmp_list.append(self.dataset[idx])
        return tmp_list

    def __len__(self):
        return len(self.dataset)
        
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)

def nt_xent(x, t=0.5, weight=None):
    # print("device of x is {}".format(x.device))
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x)

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def JS_loss(P, Q, reduction='mean'):
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    M = 0.5 * (P + Q)
    return 0.5 * (kld(P, M) + kld(Q, M))

def JS_softmax_loss(P, Q, reduction='mean'):
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    P = F.softmax(P)
    Q = F.softmax(Q)
    M = 0.5 * (P + Q)
    return 0.5 * (kld(P, M) + kld(Q, M))

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
    
def JS_normalize_loss(P, Q, reduction='mean'):
    P = normalize(P) + 1e-8
    Q = normalize(Q) + 1e-8
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    M = 0.5 * (P + Q)
    return 0.5 * (kld(P, M) + kld(Q, M))

def cos_loss(P,Q, reduction='mean'):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    if reduction == 'mean':
        return -cos(P, Q).mean()
    else:
        return -cos(P, Q)

def cos_softmax_loss(P,Q, reduction='mean'):
    P = F.softmax(P)
    Q = F.softmax(Q)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    if reduction == 'mean':
        return -cos(P, Q).mean()
    else:
        return -cos(P, Q)

def cos_normalize_loss(P,Q, reduction='mean'):
    P = normalize(P)
    Q = normalize(Q)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    if reduction == 'mean':
        return -cos(P, Q).mean()
    else:
        return -cos(P, Q)

import ot

def ot_loss(P, Q, reduction='sum'):
    batch_size = P.size(0)
    m = batch_size
    n = batch_size
    loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, F.softmax(P,dim=1),F.softmax(Q,dim=1), None, None, 0.01, m, n)
    return loss

def ot_normalize_loss(P, Q, reduction='sum'):
    P = normalize(P)
    Q = normalize(Q)
    batch_size = P.size(0)
    m = batch_size
    n = batch_size
    loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, P,Q, None, None, 0.01, m, n)
    return loss                 

# def kl_loss(P, Q, reduction='mean'):
    # kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    # return kld(F.log_softmax(P),F.softmax(Q))

def kl_loss(P, Q, reduction='mean'):
    P = F.log_softmax(P, dim=1)
    Q = F.softmax(Q, dim=1)
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    return kld(P,Q)

def mse_loss(P,Q, reduction='mean'):
    return torch.nn.MSELoss(reduction=reduction)(P, Q)


def mse_softmax_loss(P,Q, reduction='mean'):
    P = F.softmax(P)
    Q = F.softmax(Q)
    return torch.nn.MSELoss(reduction=reduction)(P, Q)

def mse_normalize_loss(P,Q, reduction='mean'):
    P = normalize(P)
    Q = normalize(Q)
    return torch.nn.MSELoss(reduction=reduction)(P, Q)

def PGD_JS(model, data, epsilon, step_size, num_steps, loss_type='JS'):
    model.eval()
    x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
    # x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
    nat_feature = model(data).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        output_feature = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_type == 'JS':
                loss_adv = JS_loss(nat_feature, output_feature)
            elif loss_type == 'JS_softmax':
                loss_adv = JS_softmax_loss(nat_feature, output_feature)
            elif loss_type == 'cos_softmax':
                loss_adv = cos_softmax_loss(nat_feature, output_feature)
            elif loss_type == 'mse_softmax':
                loss_adv = mse_softmax_loss(nat_feature, output_feature)
            elif loss_type == 'cos':
                loss_adv = cos_loss(nat_feature, output_feature)
            elif loss_type == 'mse':
                loss_adv = mse_loss(nat_feature, output_feature)
            elif loss_type == 'KL':
                loss_adv = kl_loss(output_feature,nat_feature)
            elif loss_type == 'JS_normalize':
                loss_adv = JS_normalize_loss(nat_feature, output_feature)
            elif loss_type == 'cos_normalize':
                loss_adv = cos_normalize_loss(nat_feature, output_feature)
            elif loss_type == 'mse_normalize':
                loss_adv = mse_normalize_loss(nat_feature, output_feature)
            elif loss_type == 'ot_normalize':
                loss_adv = ot_normalize_loss(nat_feature, output_feature)
            elif loss_type == 'ot':
                loss_adv = ot_loss(nat_feature, output_feature)
        # print(loss_adv)
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

class Coreset:
    def __init__(self, full_data, fraction, log, args) -> None:
        super(Coreset, self).__init__()
        self.dataset = full_data
        self.len_full = len(full_data)
        self.fraction = fraction
        self.budget = int(self.len_full * self.fraction)
        self.subset_indices = None
        self.subset_loader = None
        self.log = log
        self.args = args
    
    def update_subset_indice(self):
        pass 

    def get_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.log.info('begin subset selection')
        starttime = datetime.datetime.now()
        self.subset_indices = self.update_subset_indice()
        self.subset_loader = DataLoader(Subset(self.dataset, self.subset_indices), num_workers=4,batch_size=self.args.subset_bs,shuffle=True, pin_memory=True)
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        self.log.info('finish subset selection. subset train number: {} \t spent time: {}s'.format(len(self.subset_indices), time))
        return self.subset_loader

class RandomCoreset(Coreset):
    def __init__(self, full_data, fraction, model, log, args, online=False) -> None:
        super().__init__(full_data, fraction, log, args)
        self.lr = None
        self.model = model
        self.indices = np.random.choice(self.len_full, size=self.budget, replace=False)
        self.online = online

    def update_subset_indice(self):
        np.random.seed(self.args.seed)
        if self.online:
            self.indices = np.random.choice(self.len_full, size=self.budget, replace=False)
        return self.indices

class LossCoreset(Coreset):
    def __init__(self, full_data, fraction, log,  args, validation_loader, model) -> None:
        super().__init__(full_data, fraction, log, args)
        self.validation_loader = validation_loader
        self.model = model
        self.lr = 0.001
        if self.args.CoresetLoss == 'JS':
            self.loss_fn = JS_loss
        elif self.args.CoresetLoss == 'JS_softmax':
            self.loss_fn = JS_softmax_loss
        elif self.args.CoresetLoss == 'cos_softmax':
            self.loss_fn = cos_softmax_loss
        elif self.args.CoresetLoss == 'mse_softmax':
            self.loss_fn = mse_softmax_loss
        elif self.args.CoresetLoss == 'cos':
            self.loss_fn = cos_loss
        elif self.args.CoresetLoss == 'mse':
            self.loss_fn = mse_loss
        elif self.args.CoresetLoss == 'KL':
            self.loss_fn = kl_loss
        elif self.args.CoresetLoss == 'JS_normalize':
            self.loss_fn = JS_normalize_loss
        elif self.args.CoresetLoss == 'cos_normalize':
            self.loss_fn = cos_normalize_loss
        elif self.args.CoresetLoss == 'mse_normalize':
            self.loss_fn = mse_normalize_loss
        elif self.args.CoresetLoss == 'ot_normalize':
            self.loss_fn = ot_normalize_loss
        elif self.args.CoresetLoss == 'ot':
            self.loss_fn = ot_loss
            
    def update_subset_indice(self):
        self.log.info('use {} loss'.format(self.args.CoresetLoss))

        feature_val_nat = None
        feature_val_adv = None
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        linear_layer = self.model.module.fc
        state_dict_linear = linear_layer.state_dict()

        for name,param in linear_layer.named_parameters():
            param.requires_grad = True

        for i, (valid_inputs, target) in enumerate(self.validation_loader):
            valid_inputs = valid_inputs.cuda()
            valid_inputs_adv = PGD_JS(self.model, valid_inputs,epsilon=self.args.Coreset_epsilon, num_steps=self.args.Coreset_num_steps,
                                    step_size=self.args.Coreset_epsilon * (2 / self.args.Coreset_num_steps),loss_type=self.args.CoresetLoss)
            with torch.no_grad():
                features_adv_before_linear = self.model.module.get_feature(valid_inputs_adv)
                features_before_linear = self.model.module.get_feature(valid_inputs)
                if feature_val_nat is None:
                    feature_val_nat = features_before_linear.detach()
                    feature_val_adv = features_adv_before_linear.detach()
                else:
                    feature_val_nat = torch.cat([feature_val_nat, features_before_linear.detach()], dim=0)
                    feature_val_adv = torch.cat([feature_val_adv, features_adv_before_linear.detach()], dim=0)
        
        linear_layer.zero_grad()
        features = linear_layer(feature_val_nat)
        features_adv = linear_layer(feature_val_adv)
        valid_loss = self.loss_fn(features_adv,features)
        valid_loss.backward()

        valid_grad_list = []
        for name, param in linear_layer.named_parameters():
            g = param.grad
            # print(param, g)
            valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
            param.grad = None
        grad_val = torch.cat(valid_grad_list, dim=1)
        ori_grad_val = copy.deepcopy(grad_val)
        
        subset_index = []
        train_loader = DataLoader(IndexSubset(self.dataset), num_workers=4,batch_size=self.args.Coreset_bs,shuffle=True, pin_memory=True)
        
        batch_index_list = []
        per_batch_grads_list = []
        per_batch_ori_grads_list = []

        # begin to find the subset in each batch
        print(len(train_loader))
        for i, (idx, inputs) in enumerate(train_loader):
            target = inputs[1].cuda()
            inputs = inputs[0].cuda()
            inputs_adv = attack.pgd(self.model,inputs,target,epsilon=self.args.Coreset_epsilon, num_steps=self.args.Coreset_num_steps,
                                step_size=self.args.Coreset_epsilon * (2 / self.args.Coreset_num_steps),loss_fn='cent',category='Madry',rand_init=True)
            with torch.no_grad():
                features_adv_before_linear = self.model.module.get_feature(inputs_adv).detach()
            # initialize the gradient of each unlabel data
            linear_layer.zero_grad()
            features_adv = linear_layer(features_adv_before_linear)
            batch_loss = torch.nn.CrossEntropyLoss()(features_adv, target)
            batch_loss.backward()

            batch_grad_list = []
            batch_grad_ori_list = []
            for name, param in linear_layer.named_parameters():
                g = param.grad
                batch_grad_ori_list.append(g.detach())
                batch_grad_list.append(g.detach().mean(dim=0).view(1, -1))
                param.grad = None
            grad_batch = torch.cat(batch_grad_list, dim=1)

            per_batch_ori_grads_list.append(batch_grad_ori_list)
            per_batch_grads_list.append(grad_batch)
            batch_index_list.append(idx)
            if i % 10 == 0:
                print(i)

            if (i+1) % 50 == 0 or (i+1) == len(train_loader):
                print(i)
                per_batch_grads = torch.cat(per_batch_grads_list, dim=0)
                index_list = torch.LongTensor([q for q in range(len(batch_index_list))]).cuda()
                batch_num = math.ceil((self.budget / self.args.Coreset_bs) * (len(index_list) / len(train_loader)))
                print(batch_num)
                # Greedy search
                for j in range(batch_num):
                    # compute the gain function
                    grad_batch_list_curr = per_batch_grads[index_list]
                    gain = torch.matmul(grad_batch_list_curr, grad_val.reshape(-1,1)).squeeze()
                    print(gain.shape)
                    r = torch.argmax(gain, dim=0)
                    print(gain[r])
                    subset_index.extend(batch_index_list[index_list[r]])

                    if j == batch_num - 1:
                        break

                    # linear_layer.weight.data = linear_layer.weight.data - self.lr * per_batch_ori_grads[index_list[r]][0]
                    # linear_layer.bias.data = linear_layer.bias.data - self.lr * per_batch_ori_grads[index_list[r]][1]
                    
                    self.model.module.fc.weight.data = self.model.module.fc.weight.data - self.lr * per_batch_ori_grads_list[index_list[r]][0]
                    self.model.module.fc.bias.data = self.model.module.fc.bias.data - self.lr * per_batch_ori_grads_list[index_list[r]][1]
                    
                    # feature_val_nat = None
                    # feature_val_adv = None
                    
                    # for i, (valid_inputs, target) in enumerate(self.validation_loader):
                    #     valid_inputs = valid_inputs.cuda()
                    #     valid_inputs_adv = PGD_JS(self.model, valid_inputs,epsilon=self.args.Coreset_epsilon, num_steps=self.args.Coreset_num_steps,
                    #                             step_size=self.args.Coreset_epsilon * (2 / self.args.Coreset_num_steps),loss_type=self.args.CoresetLoss)
                    #     with torch.no_grad():
                    #         features_adv_before_linear = self.model.module.get_feature(valid_inputs_adv)
                    #         features_before_linear = self.model.module.get_feature(valid_inputs)
                    #         if feature_val_nat is None:
                    #             feature_val_nat = features_before_linear.detach()
                    #             feature_val_adv = features_adv_before_linear.detach()
                    #         else:
                    #             feature_val_nat = torch.cat([feature_val_nat, features_before_linear.detach()], dim=0)
                    #             feature_val_adv = torch.cat([feature_val_adv, features_adv_before_linear.detach()], dim=0)

                    # update grad_val with the new parameter
                    linear_layer = self.model.module.fc
                    linear_layer.zero_grad()
                    features = linear_layer(feature_val_nat)
                    features_adv = linear_layer(feature_val_adv)
                    valid_loss = self.loss_fn(features_adv,features)

                    print(valid_loss)
                    valid_loss.backward()
                    valid_grad_list = []
                    for name,param in linear_layer.named_parameters():
                        g = param.grad
                        valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
                        param.grad = None

                    grad_val = torch.cat(valid_grad_list, dim=1)
                    index_list = del_tensor_ele(index_list, r)

                # linear_layer.load_state_dict(state_dict_linear)
                self.model.module.fc.load_state_dict(state_dict_linear)
                linear_layer = self.model.module.fc
                
                # del per_batch_grads
                # del per_batch_ori_grads

                batch_index_list = []
                per_batch_grads_list = []
                per_batch_ori_grads_list = []
                grad_val = copy.deepcopy(ori_grad_val)

                # torch.cuda.empty_cache()

                print(len(subset_index), len(subset_index)/self.len_full)

        # print(np.sort(subset_index)[:1000])
        # print(np.sort(np.random.choice(self.len_full, size=self.budget, replace=False))[:1000])
        # print()
        # self.model.linear.load_state_dict(state_dict)
        return subset_index
