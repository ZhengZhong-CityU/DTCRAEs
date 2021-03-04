# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:25:29 2019

@author: zhengzhong
"""
import torch.nn.functional as F
import torch
from TCRAE_model import TCRAE
import numpy as np
from utils import data_generator
import os
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import itertools

class DTCRAE():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        print('current config: {}'.format(self.model_dir))
        self.build()
        
    @property
    def model_dir(self):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.num_channels, self.args.num_hidden, self.args.kernel_size, self.args.dropout, self.args.c_dropout, self.args.r_dropout, self.args.permute,
                self.args.batch_size, self.args.c_lr, self.args.r_lr, self.args.patience, self.args.clip, self.args.noisy_rate, self.args.c_epochs, self.args.r_epochs)
        
    def build(self):
        # do some initialization
        self.TCRAE=TCRAE(self.device, self.args.num_inputs, self.args.num_channels, self.args.num_hidden, 
                         self.args.num_classes, self.args.kernel_size, self.args.dropout, self.args.c_dropout, self.args.r_dropout)
        self.TCRAE=self.TCRAE.to(self.device)
        
        #load the data
        self.train_loader, self.val_loader, self.test_loader=self.load_data()
        
        if self.args.permute:
            self.permute=torch.Tensor(np.random.permutation(self.args.seq_length).astype(np.float64)).long().to(self.device)
        
    
    def c_train(self):
        #classification train
        print('classification training start!')
        s_time=time.time()
        optimizer = getattr(optim, self.args.optim)(self.TCRAE.parameters(), lr=self.args.c_lr)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=self.args.patience,verbose=True)     
        path=os.path.join(self.args.save_dir,self.model_dir,self.args.c_train_losses_dir)
        c_train_losses=[]
        c_val_losses=[]
        c_val_accuracy=[]
        save_figure=False
        for epoch in range(self.args.c_epochs):
            self.TCRAE.train()
            for batch_id, (data, target) in enumerate(self.train_loader):
                data = data.view(-1, self.args.num_inputs, self.args.seq_length)
#                noisy_data=self.add_noise(data)
                data=data.to(self.device)
#                noisy_data=noisy_data.to(self.device)
                target=target.to(self.device)
                if self.args.permute:
                    data=data[:, :, self.permute]
#                    noisy_data=noisy_data[:, :, permute]
                optimizer.zero_grad()
#                c_output, r_output=self.TCRAE(noisy_data, hc)
                c_output, r_output=self.TCRAE(data)
                train_loss = F.nll_loss(c_output, target)
                train_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.TCRAE.parameters(), self.args.clip)
                optimizer.step()
                e_time=time.time()
                print('batch_id|total_batch|epoch|total_epoch: [{}|{}|{}|{}] train_loss: {} time: {}'.format(batch_id, len(self.train_loader), epoch, self.args.c_epochs, train_loss.data.item(),e_time-s_time))
                s_time=time.time()
#                break
            #reduce the learning rate    
            scheduler.step(train_loss,epoch) 
            c_train_losses.append(train_loss.data.item())
           
            #evaluation
            val_loss, val_accuracy,_,_=self.c_eval(self.val_loader)
            c_val_losses.append(val_loss)
            c_val_accuracy.append(val_accuracy)
            
            if epoch==self.args.c_epochs-1:
                save_figure=True
            self.save_train_losses(c_train_losses, c_val_losses, path, epoch, self.args.c_epochs, c_val_accuracy, save_figure=save_figure)
#            break
           
        return c_train_losses, c_val_losses, c_val_accuracy
            
                     
    def c_eval(self, source):
        #classification validation or test
        print('classfication evaluation start!')
        self.TCRAE.eval()
        eval_loss = 0
        correct = 0
        length=0
        y_true=[]
        y_pred=[]
        s_time=time.time()
        with torch.no_grad():
            for data, target in source:
                data = data.view(-1, self.args.num_inputs, self.args.seq_length).to(self.device)
                target=target.to(self.device)
                if self.args.permute:
                    data = data[:, :, self.permute]
                c_output, r_output=self.TCRAE(data)
                y_true.append(target.data.cpu().numpy())
                y_pred.append(c_output.data.cpu().numpy())
                eval_loss += F.nll_loss(c_output, target, size_average=False).data.item()
                pred = c_output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().data.item()
                length+=data.size(0)
#                break
            y_true=np.concatenate(y_true)
            y_pred=np.concatenate(y_pred)          
            eval_loss /= len(source.dataset) 
            accuracy=correct/length
            e_time=time.time()
            print('classfication eval_loss: {} eval_accuracy: {} time: {}'.format(eval_loss, accuracy, e_time-s_time))
            return eval_loss, accuracy, y_true, y_pred
            
    
    def c_test(self):
        test_loss, accuracy, y_true, y_pred = self.c_eval(self.test_loader)
        path=os.path.join(self.args.save_dir,self.model_dir,self.args.c_test_losses_dir)
        self.save_test_loss(test_loss, path, accuracy)
        return test_loss, accuracy, y_true, y_pred 
    
    def r_train(self):
        #reconstruction train
        print('reconstruction training start!')
        s_time=time.time()
        optimizer = getattr(optim, self.args.optim)(self.TCRAE.parameters(), lr=self.args.r_lr)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=self.args.patience,verbose=True)     
        path=os.path.join(self.args.save_dir,self.model_dir,self.args.r_train_losses_dir)
        r_train_losses=[]
        r_val_losses=[]
        save_figure=False
        for epoch in range(self.args.r_epochs):
            self.TCRAE.train()
            for batch_id, (data, target) in enumerate(self.train_loader):
                data = data.view(-1, self.args.num_inputs, self.args.seq_length)
                noisy_data=self.add_noise(data)
                data=data.to(self.device)
                noisy_data=noisy_data.to(self.device)
                if self.args.permute:
                    data=data[:, :, self.permute]
                    noisy_data=noisy_data[:, :, self.permute]
                optimizer.zero_grad()
                c_output, r_output=self.TCRAE(noisy_data)
#                r_output.register_hook(lambda x: print(x))
                train_loss = F.mse_loss(r_output, data)
                assert r_output.shape==data.shape
#                train_loss=torch.mean((r_output-data).pow(2))
                train_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.TCRAE.parameters(), self.args.clip)
                optimizer.step()
                e_time=time.time()
                print('batch_id|total_batch|epoch|total_epoch: [{}|{}|{}|{}] train_loss: {} time: {}'.format(batch_id, len(self.train_loader), epoch, self.args.r_epochs, train_loss.data.item(),e_time-s_time))
                s_time=time.time()
#                break
#                if batch_id==2:
#                    break
            #reduce the learning rate    
            scheduler.step(train_loss,epoch) 
            r_train_losses.append(train_loss.data.item())
           
            #evaluation
            val_loss=self.r_eval(self.val_loader)
            r_val_losses.append(val_loss)
            
            if epoch==self.args.r_epochs-1:
                save_figure=True
            self.save_train_losses(r_train_losses, r_val_losses, path, epoch, self.args.r_epochs, save_figure=save_figure)
#            break
            
#            if epoch==2:
#                break
            
        return r_train_losses, r_val_losses
    
    def r_eval(self, source):
        #reconstruction validation or test       
        print('reconstruction evaluation start!')
        self.TCRAE.eval()
        eval_loss = 0
        s_time=time.time()
        with torch.no_grad():
            for data, target in source:
                data = data.view(-1, self.args.num_inputs, self.args.seq_length).to(self.device)
                if self.args.permute:
                    data = data[:, :, self.permute]
                c_output, r_output=self.TCRAE(data)
                eval_loss += F.mse_loss(r_output, data, size_average=False).data.item()
#                break
#                eval_loss=torch.sum((r_output-data).pow(2)).data.item()
#                break
#            eval_loss /= (data.shape[0]*self.args.seq_length*self.args.ninp)          
            eval_loss /= (len(source.dataset)* self.args.seq_length*self.args.num_inputs)
            e_time=time.time()
            print('reconstruction eval_loss: {}  time: {}'.format(eval_loss, e_time-s_time))
            return eval_loss
        
        
    def r_test(self):
        test_loss= self.r_eval(self.test_loader)
        path=os.path.join(self.args.save_dir,self.model_dir,self.args.r_test_losses_dir)
        self.save_test_loss(test_loss, path)
        return test_loss
    
    def save_train_losses(self, train_losses, val_losses, path, epoch, total_epochs , accuracy=None, save_figure=False):
        # save the train losses of reconstruction or classificcation
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        if save_figure:
        #save loss figure
            fig_path=os.path.join(path,'train_losses.png')
            fig=plt.figure(figsize=(12,8))
            plt.plot(train_losses, 'b', label='train_losses')
            plt.plot(val_losses,'r',label='val_losses')
            plt.xlabel('epoch')       
            if 'c_train_losses' in path:
                label='Classification loss'
            else:
                label='Reconstruction loss'         
            plt.ylabel(label)
            plt.legend()
            fig.savefig(fig_path)
            
        loss_path=os.path.join(path, 'train_losses.txt')
        with open(loss_path, 'w') as file:
            file.write('{}/{}'.format(epoch, total_epochs))
            file.write('\n')
            file.write('train_losses\n')
            file.write(str(train_losses))
            file.write('\n')
            file.write('val_losses\n')
            file.write(str(val_losses))
            file.write('\n')
            if accuracy:
               file.write('val_accuracy\n')
               file.write(str(accuracy))
        print('train losses saved to {}'.format(path))
               
    def save_test_loss(self, test_loss, path, accuracy=None):
        #save the test loss of reconstruction or classification
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        loss_path=os.path.join(path, 'test_losses.txt')
        with open(loss_path, 'w') as file:
            file.write('test_loss\n')
            file.write(str(test_loss))
            file.write('\n')
            if accuracy:
                file.write('test_accuracy\n')
                file.write(str(accuracy))
    
    def add_noise(self, samples):
        # add zero noise to the original input
        noisy_samples=samples.clone()
        for i in range(noisy_samples.shape[0]):
            flatten_sample=noisy_samples[i].reshape(-1)
            noisy_fraction=int(len(flatten_sample)*self.args.noisy_rate)
            noisy_index=np.random.choice(len(flatten_sample),noisy_fraction,replace=False)
            flatten_sample[noisy_index]=0
            noisy_sample=flatten_sample.reshape(samples[i].shape)
            noisy_samples[i]=noisy_sample
        return noisy_samples
        
    
    def load_data(self):
        # load the train, validation and test dataset
        return data_generator(self.args.root, self.args.batch_size, self.args.train_p)
    
    def save_model(self):
        #save the model
        path=os.path.join(self.args.save_dir,self.model_dir, self.args.save_model_dir)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        model_path=os.path.join(path, self.model_dir+'.pt')
        torch.save(self.TCRAE, model_path)
        print('model saved to {}!'.format(model_path))
        
    def load_model(self, model_dir):       
        model_path=os.path.join(self.args.save_dir,model_dir, self.args.save_model_dir,model_dir+'.pt')
#        root_file=os.path.join(self.args.save_dir,model_dir, self.args.save_model_dir)
#        print(os.path.exists(root_file))
#        for file in os.listdir(os.path.join(self.args.save_dir,model_dir, self.args.save_model_dir)):
#            print(file)
#            
#        target=os.path.join(root_file,'123.pt')
#        print(os.path.exists(target))
        
#        self.TCRAE=torch.load(target, map_location=self.device)
        self.TCRAE=torch.load(model_path)
        print('model loaded from {}'.format(model_path))
    

    
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Denoising TCRAE for MNIST and CIFAR10')
    parser.add_argument('--num_inputs', type=int, default=1, 
                        help='the input channels, default=1')
    parser.add_argument('--num_channels', nargs='+', type=int, default=[64]*8, 
                        help='number of hidden units per layer, default=20')
    parser.add_argument('--num_hidden', type=int, default=150, 
                        help='number of hidden units of trellisnet output, default=100')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classfication classes, default=10')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='conv layers kernel size, default=2')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='output locked dropout (0 = no dropout)')
    parser.add_argument('--c_dropout', type=float, default=0.0,
                        help='input locked dropout (0 = no dropout)')
    parser.add_argument('--r_dropout', type=float, default=0.0,
                        help='dropout applied to hidden layers (0 = no dropout)')
    parser.add_argument('--CIFAR', action='store_true', 
                        help='use CIFAR10 dataset, default=False')
    parser.add_argument('--save_dir', type=str, default='MNIST',
                        help='file dir to save the results, default=MNIST')
    parser.add_argument('--root', type=str, default='data/MNIST', 
                        help='root to save the data')
    parser.add_argument('--seq_length', type=int, default=28*28, 
                        help='length of the sequence length, 32*32 if CIFAR10')
    parser.add_argument('--permute', action='store_true', 
                        help='whether to permute the sequence, default=false')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--train_p', type=float, default=0.9, 
                        help='training percentage to split the training into training anda validation datasets')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--r_lr', type=float, default=1e-4,
                        help='learning rate for reconstruction')
    parser.add_argument('--r_epochs', type=int, default=300,
                        help='training epochs for reconstruction')
    parser.add_argument('--c_lr', type=float, default=1e-3,
                        help='learning rate for classification')
    parser.add_argument('--c_epochs', type=int, default=50,
                        help='training epochs for classification')
    parser.add_argument('--patience', type=int, default=10,
                        help='number of epochs of learning rate decay if no improvement')
    parser.add_argument('--optim', type=str, default='Adam', 
                        help='optimizer to use')
    parser.add_argument('--model_save_dir', type=str, default='saved_model',
                        help='dir to save the model')
    parser.add_argument('--r_train_losses_dir', type=str, default='r_train_losses',
                        help='dir to save reconstruction train losses')
    parser.add_argument('--r_test_losses_dir', type=str, default='r_test_losses',
                        help='dir to save reconstruction test losses')    
    parser.add_argument('--c_train_losses_dir', type=str, default='c_train_losses',
                        help='dir to save classification train losses')
    parser.add_argument('--c_test_losses_dir', type=str, default='c_test_losses',
                        help='dir to save classification test losses')     
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clipping')
    parser.add_argument('--noisy_rate', type=float, default=0.0,
                        help='noisy_rate')
    parser.add_argument('--save_model_dir', type=str, default='saved_model', 
                        help='model dir to save model')
    parser.add_argument('--pretrain', action='store_true', 
                        help='whether using reconstruction train, default: false')
    args=parser.parse_args()


    if args.permute:
        args.save_dir='MNIST_P'
#    args.CIFAR=True
#    args.permute=True
# decide MNIST or CIFAR10        
    if args.CIFAR:
        args.root='data/CIFAR10'
        args.seq_length=32*32
        args.num_inputs=3
        args.save_dir='CIFAR10'        
       
    if args.pretrain:
        args.save_dir=os.path.join(args.save_dir, 'pretrain')
    else:
        args.save_dir=os.path.join(args.save_dir, 'no_pretrain')
    
    #specify the seed and device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
     
    #grid search
    r_lr_list=[1e-4]   
    noisy_rate_list=[0.0]    
    dropout_list=[0.0,0.1,0.2]
    batch_size_list=[32, 64]

    if args.pretrain:
#        r_lr_list=[1e-4, 2e-4]
        noisy_rate_list=[0.0,0.1,0.2,0.3]

        
    best_r_lr=1e-4
    best_noisy_rate=0.0
    best_dropout=0.0
    best_batch_size=32
    best_c_val_loss=100000
    
    ############for roc use############
    # noisy_rate_list=[0.0]
    # dropout_list=[0.0,0.1,0.2]
    # batch_size_list=[32, 64]

    ############for roc use############
    
    for batch_size, r_lr, noisy_rate, dropout in itertools.product(batch_size_list, r_lr_list, noisy_rate_list, dropout_list):
        args.r_lr=r_lr
        args.noisy_rate=noisy_rate
        args.batch_size=batch_size
        args.dropout=dropout
        
        
        model=DTCRAE(args, device)
        
        ############for roc use############
        # model.load_model(model.model_dir)
        # c_test_loss, accuracy, y_true, y_pred=model.c_test()
        # print(y_true.shape)
        # print(y_pred.shape)
        
        ############for roc use############
        
        if args.pretrain:     
           r_train_losses, r_val_losses=model.r_train()
           r_test_loss=model.r_test()
 
        c_train_losses, c_val_losses, c_val_accuracy=model.c_train()
        c_test_loss, accuracy, y_true, y_pred=model.c_test()   
        model.save_model()
        
        if c_val_losses[-1]<best_c_val_loss:
           best_c_val_loss=c_val_losses[-1]
           best_model_dir=model.model_dir
               
    with open(os.path.join(args.save_dir, 'best_parameters.txt'), 'w') as file:
       file.writelines(best_model_dir)
        
        
    
    
