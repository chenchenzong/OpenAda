from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar80
import dataloader_cifar100

import copy

from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='asym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=80, type=int)
parser.add_argument('--data_path', default='./cifar-100', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp

    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                    + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                        + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def init_partition_step1(noisy_labels, num_class, K1=250, ratio1=6, ratio2=0.6, flag=False):
    CLIP_dists = torch.load('./features/CLIP/cifar100_dists.pt')

    CLIP_dists[torch.arange(CLIP_dists.size()[0]), torch.arange(CLIP_dists.size()[0])] = -1
    _, top_k_index = CLIP_dists.topk(K1, dim=1, largest=True, sorted=True)
    top_k_index = top_k_index.cpu().numpy()
    CLIP_dists = CLIP_dists.cpu()

    pesudo_targets = torch.zeros(len(noisy_labels), num_class).cuda()

    given_labels = noisy_labels
    given_targets = torch.zeros_like(pesudo_targets).cuda().scatter_(1, torch.tensor(given_labels).cuda().view(-1,1), 1)  
                
    new_targets = torch.zeros_like(pesudo_targets).cuda()
    for i in range(K1):
        new_targets += given_targets[top_k_index[:,i]]
    inference_labels = torch.argmax(new_targets, dim=1)
    inference_targets = torch.zeros_like(pesudo_targets).cuda().scatter_(1, inference_labels.view(-1,1), 1)  

    new_targets = torch.zeros_like(pesudo_targets).cuda()  
    for i in range(K1):
        new_targets += inference_targets[top_k_index[:,i]]
    inference_labels = torch.argmax(new_targets, dim=1)

    confident_clean_threshold = ratio1*K1/num_class 
    given_label_logits = new_targets[range(len(noisy_labels)),noisy_labels]

    confident_clean_indexes = np.arange(len(noisy_labels))[given_label_logits.cpu().numpy()>=confident_clean_threshold].tolist()

    count_by_class = [0 for i in range(num_class)]
    for index in range(len(noisy_labels)):
        label = noisy_labels[index]
        count_by_class[label] += 1

    count_by_class = [0 for i in range(num_class)]
    for index in confident_clean_indexes:
        label = noisy_labels[index]
        count_by_class[label] += 1
    count_mean = int(np.mean(count_by_class))

    count_by_class = count_mean - np.asarray(count_by_class)
    count_by_class[count_by_class<0] = 0

    copy_new_targets = copy.deepcopy(new_targets)*given_targets
    for index in confident_clean_indexes:
        copy_new_targets[index, :] = -1e5
    for class_index in range(num_class):
        if count_by_class[class_index] > 0:
            _, selected_index = copy_new_targets[:,class_index].topk(count_by_class[class_index], largest=True, sorted=True)
            confident_clean_indexes.extend(selected_index.cpu().numpy())
            confident_clean_indexes = list(set(confident_clean_indexes))

    count_by_class = [0 for i in range(num_class)]
    for index in confident_clean_indexes:
        label = noisy_labels[index]
        count_by_class[label] += 1

    score_assign = np.zeros(len(noisy_labels))
    score_assign[confident_clean_indexes] = 1

    confident_count = np.zeros(len(noisy_labels))
    for i in range(K1):
        confident_count += score_assign[top_k_index[:,i]]

    confident_open_threshold = ratio2*K1*len(confident_clean_indexes)/len(noisy_labels)

    confident_count = np.asarray(confident_count)
    confident_count[confident_clean_indexes] = 1e5
    confident_open_indexes = np.arange(len(noisy_labels))[confident_count<confident_open_threshold].tolist()

    torch.cuda.empty_cache()
    return confident_clean_indexes, confident_open_indexes, top_k_index


def init_partition_step2(noisy_labels, confident_clean_indexes, top_k_index, K1=250, ratio2=0.6):

    score_assign = np.zeros(len(noisy_labels))
    score_assign[confident_clean_indexes] = 1
    confident_count = np.zeros(len(noisy_labels))
    for i in range(K1):
        confident_count += score_assign[top_k_index[:,i]]

    confident_open_threshold = ratio2*K1*len(confident_clean_indexes)/len(noisy_labels)

    confident_count = np.asarray(confident_count)
    confident_count[confident_clean_indexes] = 1e5
    confident_open_indexes = np.arange(len(noisy_labels))[confident_count<confident_open_threshold].tolist()

    torch.cuda.empty_cache()
    return confident_open_indexes


def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class+1).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            _, outputs_open_u11 = net(inputs_u)
            _, outputs_open_u12 = net(inputs_u2)
            _, outputs_open_u21 = net2(inputs_u)
            _, outputs_open_u22 = net2(inputs_u2) 

            outputs_open_u11 = F.softmax(outputs_open_u11.view(outputs_open_u11.size(0), 2, -1), 1)[:,1,:].detach() 
            outputs_open_u12 = F.softmax(outputs_open_u12.view(outputs_open_u12.size(0), 2, -1), 1)[:,1,:].detach() 
            outputs_open_u21 = F.softmax(outputs_open_u21.view(outputs_open_u21.size(0), 2, -1), 1)[:,1,:].detach() 
            outputs_open_u22 = F.softmax(outputs_open_u22.view(outputs_open_u22.size(0), 2, -1), 1)[:,1,:].detach()           
            
            pu = (outputs_open_u11 + outputs_open_u12 + outputs_open_u21 + outputs_open_u22) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach() 

            targets_open_u = pu.detach()
            
            # label refinement of labeled samples
            outputs_x, _ = net(inputs_x)
            outputs_x2, _ = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        all_targets_open = torch.cat([labels_x, labels_x, targets_open_u, targets_open_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        target_open_a, target_open_b = all_targets_open, all_targets_open[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        mixed_target_open = l * target_open_a + (1 - l) * target_open_b
                
        logits, _ = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]       
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        

        _, logits_open = net(all_inputs)
        logits_open_x = logits_open[:batch_size*2]
        logits_open_u = logits_open[batch_size*2:] 

        Lx_open = ova_loss(logits_open_x, all_targets_open[:batch_size*2].max(1)[1])
        
        logits_open_u = logits_open_u.view(logits_open_u.size(0), 2, -1)
        logits_open_u = F.softmax(logits_open_u, 1)
        Lu_open = 0.5*torch.mean(torch.sum(torch.abs(all_targets_open[batch_size*2:] - logits_open_u[:,1,:])**2 + torch.abs((1-all_targets_open[batch_size*2:]) - logits_open_u[:,0,:])**2,1))

        # regularization
        prior = torch.ones(args.num_class+1)/(args.num_class+1)
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))


        loss = Lx+Lx_open + Lu_open + lamb*Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader,extra_dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):  
        try:
            inputs_extra, labels_extra, _ = extra_dataloader_iter.next()
        except:
            extra_dataloader_iter = iter(extra_dataloader)
            inputs_extra, labels_extra, _ = extra_dataloader_iter.next() 
        
        inputs, labels = inputs.cuda(), labels.cuda() 
        inputs_extra, labels_extra = inputs_extra.cuda(), labels_extra.cuda() 
        
        optimizer.zero_grad()
        outputs, outputs_open = net(inputs) 
        outputs_extra, outputs_extra_open = net(inputs_extra) 

        loss = CEloss(outputs, labels)  
        loss += CEloss(outputs_extra, labels_extra)     
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)+conf_penalty(outputs_extra)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss

        L += ova_loss(outputs_open, labels)

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs)
            outputs1 = outputs1[:,:-1]
            outputs2, _ = net2(inputs)  
            outputs2 = outputs2[:,:-1]         
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def eval_train(basic_noisy_labels, confident_clean_indexes, confident_open_indexes, top_k_index, model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)  
    ova_score = torch.zeros(50000)  
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            
            outputs, outputs_open = model(inputs) 
            outputs_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)[:,1,:]
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b] 
                ova_score[index[b]] = outputs_open[b,targets[b]].item()

    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    ova_score = ova_score.numpy().reshape(-1,1)
    th = threshold_otsu(ova_score)

    ova_score[ova_score>=th] = 1
    ova_score[ova_score<th] = 0
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()] 
    prob = prob*(ova_score.squeeze())

    clean_indexes = np.where(prob > 0.5)[0]
    union_clean_indexes = list(set(clean_indexes.tolist()) | set(confident_clean_indexes))
    open_indexes = init_partition_step2(basic_noisy_labels, union_clean_indexes, top_k_index)
    union_open_indexes = list(set(open_indexes) | set(confident_open_indexes))
    prob[confident_clean_indexes] = 1
    prob[union_open_indexes] = 2

    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = model = ResNet18(num_classes=args.num_class+1)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/%s_%s_%.1f_%s'%(args.dataset,args.num_class,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%s_%.1f_%s'%(args.dataset,args.num_class,args.r,args.noise_mode)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 10

if args.num_class == 80:
    loader = dataloader_cifar80.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))
elif args.num_class == 100:
    loader = dataloader_cifar100.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

dataset = loader.run('eval_train').dataset
basic_noisy_labels = []
for i in range(len(dataset)):
    target = dataset.noise_label[i]  
    basic_noisy_labels.append(target)  
basic_confident_clean_indexes, basic_confident_open_indexes, top_k_index = init_partition_step1(basic_noisy_labels, args.num_class)


for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr  

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader, extra_trainloader = loader.run('warmup', confident_clean_indexes=basic_confident_clean_indexes, confident_open_indexes=basic_confident_open_indexes)
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader,extra_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader,extra_trainloader) 

        prob1,all_loss[0]=eval_train(basic_noisy_labels, basic_confident_clean_indexes, basic_confident_open_indexes, top_k_index, net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(basic_noisy_labels, basic_confident_clean_indexes, basic_confident_open_indexes, top_k_index, net2,all_loss[1])          
          
    else:         
        prob1,all_loss[0]=eval_train(basic_noisy_labels, basic_confident_clean_indexes, basic_confident_open_indexes, top_k_index, net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(basic_noisy_labels, basic_confident_clean_indexes, basic_confident_open_indexes, top_k_index, net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    test(epoch,net1,net2)  


