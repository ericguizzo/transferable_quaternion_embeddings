import sys, os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
import torch.utils.data as utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import *
from loss_emo import *
import utility_functions as uf
from tqdm import tqdm

parser = argparse.ArgumentParser()
#saving parameters
parser.add_argument('--experiment_name', type=str, default='test')
parser.add_argument('--results_folder', type=str, default='../results')
parser.add_argument('--results_path', type=str, default='../results/results.npy')
parser.add_argument('--model_path', type=str, default='../results/model')
#dataset parameters
#'../new_experiments/experiment_1_beta0.txt/models/model_xval_iemocap_exp1_beta0.txt_run1_fold0'
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--predictors_normailzation', type=str, default='01')
parser.add_argument('--fast_test', type=str, default='True')
parser.add_argument('--fast_test_bound', type=int, default=5)
parser.add_argument('--shuffle_data', type=str, default='False')

#training parameters
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--regularization_lambda', type=float, default=0.)
parser.add_argument('--early_stopping', type=str, default='True')
parser.add_argument('--save_model_metric', type=str, default='total_loss')
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=0)
parser.add_argument('--fixed_seed', type=str, default=None)
parser.add_argument('--spreadsheet_profile', type=str, default=None)

#loss parameters
parser.add_argument('--loss_function', type=str, default='emotion_recognition_loss')
parser.add_argument('--loss_beta', type=float, default=1.)
parser.add_argument('--emo_loss_holes', type=int, default=None)  #emo loss is deactivated every x epochs
parser.add_argument('--emo_loss_warmup_epochs', type=int, default=None)  #warmup ramp length


#model parameters
parser.add_argument('--model_name', type=str, default='VGGNet')
parser.add_argument('--model_architecture', type=str, default='VGG16')
parser.add_argument('--model_quat', type=str, default='True')
parser.add_argument('--model_classifier_quat', type=str, default='True')
parser.add_argument('--model_conv_structure', type=str, default='[16,32,64,128,256]')
parser.add_argument('--model_classifier_structure', type=str, default='[4096,4096]')
parser.add_argument('--model_batch_normalization', type=str, default='True')
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--model_flatten_dim', type=int, default=32768)
parser.add_argument('--model_classifier_dropout', type=float, default=0.5)
parser.add_argument('--model_num_classes', type=int, default=4)
parser.add_argument('--model_embeddings_dim', type=str, default='[64,64]')
parser.add_argument('--model_verbose', type=str, default='False')

parser.add_argument('--use_r2he', type=str, default='True')
parser.add_argument('--r2he_model_path', type=str, default=None)
parser.add_argument('--r2he_model_name', type=str, default='simple_autoencoder')
parser.add_argument('--r2he_features_type', type=str, default='reconstruction',
                    help='reconstruction or embeddings')


#grid search parameters
#SPECIFY ONLY IF PERFORMING A GRID SEARCH WITH exp_instance.py SCRIPT
parser.add_argument('--script', type=str, default='training_autoencoder.py')
parser.add_argument('--comment_1', type=str, default='none')
parser.add_argument('--comment_2', type=str, default='none')
parser.add_argument('--experiment_description', type=str, default='none')
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--num_experiment', type=int, default=0)
#        "load_pretrained": "'../new_experiments/experiment_9_5samples.txt/models/model_xval_iemocap_exp9_5samples.txt_run1_fold0'"

#eval string args
args = parser.parse_args()
#output filenames

args.fast_test = eval(args.fast_test)
args.use_cuda = eval(args.use_cuda)
args.early_stopping = eval(args.early_stopping)
args.fixed_seed = eval(args.fixed_seed)
args.model_verbose = eval(args.model_verbose)
args.model_quat = eval(args.model_quat)
args.model_classifier_quat = eval(args.model_classifier_quat)
args.model_batch_normalization = eval(args.model_batch_normalization)
args.model_conv_structure = eval(args.model_conv_structure)
args.model_classifier_structure = eval(args.model_classifier_structure)
args.model_embeddings_dim = eval(args.model_embeddings_dim)
args.shuffle_data = eval(args.shuffle_data)
args.use_r2he = eval(args.use_r2he)


if args.use_cuda:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

#load data loaders
tr_data, val_data, test_data = uf.load_datasets(args)

#load model

print ('\nMoving model to device')
if args.model_name == 'VGGNet':
    model = VGGNet(architecture=args.model_architecture,
                   classifier_dropout=args.model_classifier_dropout,
                   flatten_dim=args.model_flatten_dim,
                   verbose=args.model_verbose,
                   quat=args.model_quat,
                   num_classes=args.model_num_classes
                   )
elif args.model_name == 'AlexNet':
    model = AlexNet(quat=args.model_quat,
                    num_classes=args.model_num_classes
                    )
elif args.model_name == 'resnet50':
    model = resnet50(quat=args.model_quat,
                    num_classes=args.model_num_classes
                    )
else:
    raise ValueError('Invalid model name')

model = model.to(device)

#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

#load pretrained model if desired
if args.load_pretrained is not None:
    print ('Loading pretrained model: ' + args.load_pretrained)
    pretrained_dict = torch.load(args.load_pretrained)
    l_w = list(pretrained_dict.keys())[-1]
    l_b = list(pretrained_dict.keys())[-2]
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    del pretrained_dict[l_w]
    del pretrained_dict[l_b]
    model.load_state_dict(pretrained_dict, strict=False)  #load best model

#load r2he model if desired
if args.use_r2he:
    if args.r2he_model_name == 'simple_autoencoder':
        r2he = simple_autoencoder()
    pretrained_dict_r2he = torch.load(args.r2he_model_path)
    print ('loading r2he: ', args.r2he_model_path)
    r2he.load_state_dict(pretrained_dict_r2he, strict=False)
    r2he = r2he.to(device)


#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.regularization_lambda)

#loss_function = nn.BCELoss()
loss_function = locals()[args.loss_function]

#init history
train_loss_hist = []
val_loss_hist = []

def evaluate(model, device, loss_function, dataloader, emo_weight):
    #compute loss without backprop
    model.eval()
    temp_loss = []
    with tqdm(total=len(dataloader)) as pbar, torch.no_grad():
        #validation data
        for i, (sounds, truth) in enumerate(dataloader):
            sounds = sounds.to(device)
            truth = truth.to(device)

            #generate quaternion emotional embeddings if desired
            if args.use_r2he:
                if args.r2he_features_type == 'reconstruction':
                    sounds, _ = r2he(sounds)
                elif args.r2he_features_type == 'embeddings':
                    sounds, _ = r2he.get_embeddings(sounds)
                else:
                    raise ValueError('wrong r2he features type selected')

            pred = model(sounds)

            loss = loss_function(pred, truth)

            temp_loss.append({'loss':loss['loss'].cpu().numpy(),
                              'acc': loss['acc']})
            pbar.update(1)
    return temp_loss

def mean_batch_loss(batch_loss):
    #compute mean of each loss item
    d = {'loss':[], 'acc':[]}
    for i in batch_loss:
        for j in i:
            name = j
            value = i[j]
            d[name].append(value)
    for i in d:
        d[i] = np.mean(d[i])
    return d

#training loop
for epoch in range(args.num_epochs):
    epoch_start = time.perf_counter()
    print ('\n')
    print ('Epoch: [' + str(epoch+1) + '/' + str(args.num_epochs) + '] ')
    train_batch_losses = []
    model.train()

    #emotional loss warm up and holes
    emo_weight = args.loss_beta
    #warm up
    if args.emo_loss_warmup_epochs is not None:
        if epoch < args.emo_loss_warmup_epochs:
            ramp = np.arange(args.emo_loss_warmup_epochs) / args.emo_loss_warmup_epochs
            w = ramp[epoch]
            emo_weight = emo_weight * w
    #holesramp
    if args.emo_loss_holes is not None:
        if epoch % args.emo_loss_holes == 0:
            emo_weight = 0

    #train data
    with tqdm(total=len(tr_data)) as pbar:
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)

            #generate quaternion emotional embeddings if desired
            if args.use_r2he:
                with torch.no_grad():
                    if args.r2he_features_type == 'reconstruction':
                        sounds, _ = r2he(sounds)
                    elif args.r2he_features_type == 'embeddings':
                        sounds, _ = r2he.get_embeddings(sounds)
                    else:
                        raise ValueError('wrong r2he features type selected')
            #print (sounds.shape)
            pred = model(sounds)

            #recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1) / 4.
            #recon = torch.unsqueeze(torch.sqrt(torch.sum(recon**2, axis=1)), dim=1)
            #loss = loss_function(recon, sounds)
            loss = loss_function(pred, truth)
            loss['loss'].backward()
            optimizer.step()

            #loss = loss.detach().cpu().item()
            train_batch_losses.append({'loss':loss['loss'].detach().cpu().numpy(),
                                       'acc': loss['acc']})
            pbar.update(1)
            #del loss

    #validation data
    val_batch_losses = evaluate(model, device, loss_function, val_data, emo_weight)

    train_epoch_loss = mean_batch_loss(train_batch_losses)
    val_epoch_loss = mean_batch_loss(val_batch_losses)

    print ('\n EPOCH LOSSES:')
    print ('\n Training:')
    print (train_epoch_loss)
    print ('\n Validation:')
    print (val_epoch_loss)
    print ('Comments:')
    print (args.comment_1, args.comment_2)

    train_loss_hist.append(train_epoch_loss)
    val_loss_hist.append(val_epoch_loss)


    #compute epoch time
    epoch_time = float(time.perf_counter()) - float(epoch_start)
    print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

    #save best model (metrics = validation loss)
    if epoch == 0:
        torch.save(model.state_dict(), args.model_path)
        print ('\nModel saved')
        saved_epoch = epoch + 1
    else:
        if args.save_model_metric == 'loss':
            best_loss = min([i['loss'] for i in val_loss_hist[:-1]])
            #best_loss = min(val_loss_hist['total'].item()[:-1])  #not looking at curr_loss
            curr_loss = val_loss_hist[-1]['loss']
            if curr_loss < best_loss:
                torch.save(model.state_dict(), args.model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1
        elif args.save_model_metric == 'acc':
            best_loss = max([i['acc'] for i in val_loss_hist[:-1]])
            #best_loss = min(val_loss_hist['total'].item()[:-1])  #not looking at curr_loss
            curr_loss = val_loss_hist[-1]['acc']
            if curr_loss > best_loss:
                torch.save(model.state_dict(), args.model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1
        elif args.save_model_metric == 'epochs':
            if epoch % 100 == 0:
                torch.save(model.state_dict(), args.model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1
        else:
            raise ValueError('Wrong metric selected')

    if args.early_stopping and epoch >= args.patience+1:
        patience_vec = [i['loss'] for i in val_loss_hist[-args.patience+1:]]
        #patience_vec = val_loss_hist[-args.patience+1:]
        best_l = np.argmin(patience_vec)
        if best_l == 0:
            print ('Training early-stopped')
            break


#COMPUTE METRICS WITH BEST SAVED MODEL
print ('\nComputing metrics with best saved model')

model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model

train_batch_losses = evaluate(model, device, loss_function, tr_data, 1)
val_batch_losses = evaluate(model, device, loss_function, val_data, 1)
test_batch_losses = evaluate(model, device, loss_function, test_data, 1)

train_loss = mean_batch_loss(train_batch_losses)
val_loss = mean_batch_loss(val_batch_losses)
test_loss = mean_batch_loss(test_batch_losses)

#save results in temp dict file
temp_results = {}

#save loss
temp_results['train_loss'] = train_loss['loss']
temp_results['val_loss'] = val_loss['loss']
temp_results['test_loss'] = test_loss['loss']

temp_results['train_acc'] = train_loss['acc']
temp_results['val_acc'] = val_loss['acc']
temp_results['test_acc'] = test_loss['acc']

temp_results['train_loss_hist'] = train_loss_hist
temp_results['val_loss_hist'] = val_loss_hist
temp_results['parameters'] = vars(args)

np.save(args.results_path, temp_results)

#print  results
print ('\nRESULTS:')
keys = list(temp_results.keys())
keys.remove('parameters')
keys.remove('train_loss_hist')
keys.remove('val_loss_hist')

train_keys = [i for i in keys if 'train' in i]
val_keys = [i for i in keys if 'val' in i]
test_keys = [i for i in keys if 'test' in i]

print ('\n train:')
for i in train_keys:
    print (i, ': ', temp_results[i])
print ('\n val:')
for i in val_keys:
    print (i, ': ', temp_results[i])
print ('\n test:')
for i in test_keys:
    print (i, ': ', temp_results[i])
