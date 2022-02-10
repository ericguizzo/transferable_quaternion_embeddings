import torch
from torch import nn
import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
from models import *
import argparse
import os
from tqdm import tqdm
import utility_functions as uf
parser = argparse.ArgumentParser()

parser.add_argument('--datapoints_list', type=str, default='[1,2,3,4,5]')
parser.add_argument('--figures_path', type=str, default='../properties/NEW_experiments/figures')
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--fixed_seed', type=str, default='True')

#dataset parameters
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--use_set', type=str, default='test')
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--predictors_normailzation', type=str, default='01')
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--fast_test', type=str, default='True')
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)
#model parameters
#'../new_experiments/experiment_6_beta0_noquat.txt/models/model_xval_iemocap_exp6_beta0_noquat.txt_run1_fold0'
#'../new_experiments/experiment_4_beta0.txt/models/model_xval_iemocap_exp4_beta0.txt_run1_fold0'
#'../new_experiments/experiment_3_beta0.txt/models/model_xval_iemocap_exp3_beta0.txt_run1_fold0'
#'../new_experiments/experiment_7_beta0.txt/models/model_xval_iemocap_exp7_beta0.txt_run1_fold0'
'../new_experiments/experiment_8_beta0_simple.txt/models/model_xval_iemocap_exp8_beta0_simple.txt_run1_fold0'
'../new_experiments/experiment_9_5samples.txt/models/model_xval_iemocap_exp9_5samples.txt_run1_fold0'
'../new_experiments/experiment_11_5adding_emo_noload.txt/models/model_xval_iemocap_exp11_5adding_emo_noload.txt_run1_fold0'
'../new_experiments/experiment_12_5samples.txt/models/model_xval_iemocap_exp12_5samples.txt_run1_fold0'
'../new_experiments/experiment_13_5samples.txt/models/model_xval_iemocap_exp13_5samples.txt_run1_fold0'
'../new_experiments/experiment_10_5adding_emo.txt/models/model_xval_iemocap_exp10_5adding_emo.txt_run1_fold0'
'../new_experiments/experiment_16_betagrid.txt/models/model_xval_iemocap_exp16_betagrid.txt_run1_fold0'
'../new_experiments/experiment_15_pretraining_reconstruction.txt/models/model_xval_iemocap_exp15_pretraining_reconstruction.txt_run1_fold0'
'../new_experiments/experiment_15_pretraining_reconstruction.txt/models/model_xval_iemocap_exp15_pretraining_reconstruction.txt_run1_fold0'
'../new_experiments/experiment_323_DACCAPO_betagrid_RIPARTO.txt/models/model_xval_iemocap_exp323_DACCAPO_betagrid_RIPARTO.txt_run2_fold0'
'../new_experiments/experiment_315_DACCAPO_pretraining_reconstruction.txt/models/model_xval_iemocap_exp315_DACCAPO_pretraining_reconstruction.txt_run1_fold0'
'../new_experiments/experiment_314_DACCAPO_pretraining_reconstructionMODS.txt/models/model_xval_iemocap_exp315_DACCAPO_pretraining_reconstructionMODS.txt_run1_fold0'
'../new_experiments/experiment_314_DACCAPO_pretraining_reconstructionMODS.txt/models/model_xval_iemocap_exp314_DACCAPO_pretraining_reconstructionMODS.txt_run1_fold0'


M_PATH='../new_experiments/experiment_314_DACCAPO_pretraining_reconstructionMODS.txt/models/model_xval_iemocap_exp314_DACCAPO_pretraining_reconstructionMODS.txt_run1_fold0'



parser.add_argument('--model_path', type=str, default=M_PATH)
parser.add_argument('--model_name', type=str, default='simple_autoencoder_2_vad_mod')
parser.add_argument('--model_quat', type=str, default='True')
parser.add_argument('--model_in_channels', type=int, default=1)
parser.add_argument('--model_flattened_dim', type=int, default=524288)
parser.add_argument('--model_latent_dim', type=int, default=1000)
parser.add_argument('--model_verbose', type=str, default='False')
parser.add_argument('--model_architecture', type=str, default='simple')
parser.add_argument('--model_classifier_dropout', type=float, default=0.5)
parser.add_argument('--fast_test_bound', type=int, default=100)
parser.add_argument('--shuffle_data', type=str, default='False')
parser.add_argument('--model_hidden_size', type=int, default=4096)

args = parser.parse_args()

args.use_cuda = eval(args.use_cuda)
args.datapoints_list = eval(args.datapoints_list)
args.model_verbose = eval(args.model_verbose)
args.fast_test = eval(args.fast_test)
args.model_quat = eval(args.model_quat)
args.shuffle_data = eval(args.shuffle_data)


#def gen_plot(o, r, v, a, d, sound_id, curr_path, format='png'):
def gen_plot(sounds, pred, sound_id, args):
    #pred = pred.cpu().numpy()
    #recon = torch.unsqueeze(torch.sum(pred, axis=1), dim=1) / 4.
    #recon = recon.cpu().numpy().squeeze()
    sounds = sounds[0].cpu().numpy().squeeze()

    #pred,_,_,_ = pred
    pred = pred[0].cpu().numpy().squeeze()
    '''
    if len(pred.shape)== 3:
        r = pred[0]
        v = pred[1]
        a = pred[2]
        d = pred[3]
    else:
        r = pred
        v = pred
        a = pred
        d = pred
    '''
    r = pred[0]
    v = pred[1]
    a = pred[2]
    d = pred[3]
    #r = pred
    #v = pred
    #a = pred
    #d = pred
    print ('ajajajajaj', sounds.shape, pred.shape)
    recon = np.sum(pred, axis=0)
    #recon = np.sum(pred**2, axis=0)
    #r = np.flip(r.T,-1)
    #v = np.flip(v.T,-1)
    #a = np.flip(a.T,-1)
    #d = np.flip(d.T,-1)


    print ('max: ', np.max(recon), np.max(r),np.max(v),np.max(a),np.max(d))
    print ('mean: ', np.mean(recon), np.mean(r),np.mean(v),np.mean(a),np.mean(d))
    #exponent = 2/3
    exponent = 1/3
    '''
    r = (np.flip(r.T,-1)/np.max(r))**exponent
    v = (np.flip(v.T,-1)/np.max(v))**exponent
    a = (np.flip(a.T,-1)/np.max(a))**exponent
    d = (np.flip(d.T,-1)/np.max(d))**exponent
    sounds = (np.flip(sounds.T,-1)/np.max(sounds))**exponent
    '''

    r = (np.flip(r.T,-1))**exponent
    v = (np.flip(v.T,-1))**exponent
    a = (np.flip(a.T,-1))**exponent
    d = (np.flip(d.T,-1))**exponent
    sounds = (np.flip(sounds.T,-1))**exponent

    plt.figure(1)
    plt.suptitle('AUTOENCODER OUTPUT MATRICES')
    plt.subplot(231)
    plt.title('Output: Real')
    plt.pcolormesh(r)
    plt.subplot(232)
    plt.title('Output: i')
    plt.pcolormesh(v)
    plt.subplot(233)
    plt.title('Output: j')
    plt.pcolormesh(a)
    plt.subplot(234)
    plt.title('Output: k')
    plt.pcolormesh(d)
    plt.subplot(235)
    plt.title('Input')
    plt.pcolormesh(sounds)
    plt.subplot(236)
    if len(pred.shape)== 3:
        plt.title('Output: mean')
        plt.pcolormesh(np.flip(recon.T,-1)**exponent)

    plt.tight_layout( rect=[0, 0.0, 0.95, 0.95])

    #save fig
    if not os.path.exists(args.figures_path):
        os.makedirs(args.figures_path)
    name = str(sound_id) + '_plot.png'
    fig_name = os.path.join(args.figures_path, name)
    plt.savefig(fig_name, format = 'png', dpi=300)


if __name__ == '__main__':
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    #load data loaders
    tr_data, val_data, test_data = uf.load_datasets(args)

    if args.use_set == 'training':
        dataloader = tr_data
    elif args.use_set == 'validation':
        dataloader = val_data
    elif args.use_set == 'test':
        dataloader = test_data

    #load model
    print ('\nMoving model to device')
    if args.model_name == 'r2he':
        model = locals()[args.model_name](latent_dim=args.model_latent_dim,
                                          in_channels=args.model_in_channels,
                                          architecture=args.model_architecture,
                                          classifier_dropout=args.model_classifier_dropout,
                                          flattened_dim=args.model_flattened_dim,
                                          quat=args.model_quat,
                                          verbose=args.model_verbose)
    elif args.model_name == 'simple_autoencoder':
        print ('AAAAAFJFJFJFJFJFJFJFJFJFJFJFJ')
        model = locals()[args.model_name]()
    elif args.model_name == 'simple_autoencoder_2_vad':
        model = locals()[args.model_name](quat=args.model_quat,
                                          classifier_quat=False,
                                          hidden_size=args.model_hidden_size)
    elif args.model_name == 'simple_autoencoder_2_vad_mod':
        print ('AAAAAFJFJFJFJFJFJFJFJFJFJFJFJ')
        model = locals()[args.model_name](quat=args.model_quat,
                                          hidden_size=args.model_hidden_size,
                                          )
    else:
        raise ValueError('Invalid model name')

    model = model.to(device)

    #load pretrained model if desired
    model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model

    #iterate batches
    model.eval()
    with tqdm(total=len(args.datapoints_list)) as pbar, torch.no_grad():
        for i, (sounds, truth) in enumerate(dataloader):
            if i in args.datapoints_list:
                sounds = sounds.to(device)
                pred, _, _, _, _ = model(sounds)

                gen_plot(sounds, pred, i, args)
                pbar.update(1)
