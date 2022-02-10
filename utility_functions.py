from __future__ import print_function
import numpy as np
import math, copy
import os
import pandas
import random
from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import iirfilter, butter, filtfilt, lfilter
from shutil import copyfile
import librosa
import configparser
import time
import torch
import torch.utils.data as utils



cfg = configparser.ConfigParser()
cfg.read('preprocessing_config.ini')

FIXED_SEED = cfg.get('sampling', 'fixed_seed')
SR = cfg.getint('sampling', 'sr_target')


FIXED_SEED = eval(FIXED_SEED)

if FIXED_SEED is not None:
    # Set seed
    manualSeed = FIXED_SEED
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

tol = 1e-14    # threshold used to compute phase

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def isPower2(num):
    #taken from Xavier Serra's sms tools
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0

def wavread(file_name):
    #taken from Xavier Serra's sms tools
    '''
    read wav file and converts it from int16 to float32
    '''
    sr, samples = read(file_name)
    samples = np.float32(samples)/norm_fact[samples.dtype.name] #float conversion

    return sr, samples

def wavwrite(y, fs, filename):
    #taken from Xavier Serra's sms tools
    """
    Write a sound file from an array with the sound and the sampling rate
    y: floating point array of one dimension, fs: sampling rate
    filename: name of file to create
    """
    x = copy.deepcopy(y)                         # copy array
    x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
    x = np.int16(x)                              # converting to int16 type
    write(filename, fs, x)

def zeropad_2d(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def print_bar(index, total):
    perc = int(index / total * 20)
    perc_progress = int(np.round((float(index)/total) * 100))
    inv_perc = int(20 - perc - 1)
    strings = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
    print ('\r', strings, end='')

def folds_generator(num_folds, foldable_list, percs):
    '''
    create dict with a key for every actor (or foldable idem)
    in each key are contained which actors to put in train, val and test
    '''
    tr_perc = percs[0]
    val_perc = percs[1]
    test_perc = percs[2]
    num_actors = len(foldable_list)
    ac_list = foldable_list * num_folds

    n_train = int(np.round(num_actors * tr_perc))
    n_val = int(np.round(num_actors * val_perc))
    n_test = int(num_actors - (n_train + n_val))

    #ensure that no set has 0 actors
    if n_test == 0 or n_val == 0:
        n_test = int(np.ceil(num_actors*test_perc))
        n_val = int(np.ceil(num_actors*val_perc))
        n_train = int(num_actors - (n_val + n_test))

    shift = num_actors / num_folds
    fold_actors_list = {}
    for i in range(num_folds):
        curr_shift = int(shift * i)
        tr_ac = ac_list[curr_shift:curr_shift+n_train]
        val_ac = ac_list[curr_shift+n_train:curr_shift+n_train+n_val]
        test_ac = ac_list[curr_shift+n_train+n_val:curr_shift+n_train+n_val+n_test]
        fold_actors_list[i] = {'train': tr_ac,
                          'val': val_ac,
                          'test': test_ac}

    return fold_actors_list

def build_matrix_dataset(merged_predictors, merged_target, actors_list):
    '''
    load preprocessing dict and output numpy matrices of predictors and target
    containing only samples defined in actors_list
    '''

    predictors = []
    target = []
    index = 0
    total = len(actors_list)
    for i in actors_list:
        for j in range(merged_predictors[i].shape[0]):
            #print ('CAZZO', merged_predictors[i][j].shape)
            predictors.append(merged_predictors[i][j])
            target.append(merged_target[i][j])
        index += 1
        perc = int(index / total * 20)
        perc_progress = int(np.round((float(index)/total) * 100))
        inv_perc = int(20 - perc - 1)
        string = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
        print ('\r', string, end='')
    predictors = np.array(predictors)
    target = np.array(target)
    print(' | shape: ' + str(predictors.shape))
    print ('\n')

    return predictors, target

def find_longest_audio(input_folder):
    '''
    look for all .wav files in a folder and
    return the duration (in samples) of the longest one
    '''
    contents = os.listdir(input_folder)
    file_sizes = []
    for file in contents:
        if file[-3:] == "wav": #selects just wav files
            file_name = input_folder + '/' + file   #construct file_name string
            try:
                samples, sr = librosa.core.load(file_name, sr=SR)  #read audio file
                #samples = strip_silence(samples)
                file_sizes.append(len(samples))
            except ValueError:
                pass
    max_file_length = max(file_sizes)
    max_file_length = (max_file_length + 10 )/ float(sr)

    return max_file_length, sr

def find_longest_audio_list(input_list):
    '''
    look for all .wav files in a folder and
    return the duration (in samples) of the longest one
    '''
    file_sizes = []
    for file in input_list:
        if file[-3:] == "wav": #selects just wav files
            samples, sr = librosa.core.load(file, sr=SR)  #read audio file
            #print ('MERDAAAAAAAA', sr)

            file_sizes.append(len(samples))

    max_file_length = max(file_sizes)
    max_file_length = (max_file_length + 10 )/ float(sr)

    return max_file_length, sr

def find_longest_audio_list2(input_list):
    '''
    look for all .wav files in a folder and
    return the duration (in samples) of the longest one
    '''
    file_sizes = []
    for file in input_list:
        if file[-3:] == "wav": #selects just wav files
            samples, sr = librosa.core.load(file, sr=48000)  #read audio file
            print ('MERDAAAAAAAA', sr)

            file_sizes.append(len(samples))

    max_file_length = max(file_sizes)
    max_file_length = (max_file_length + 10 )/ float(sr)

    mean = np.mean(file_sizes)
    std = np.std(file_sizes)
    print ('mean', int(mean * sr))
    print ('std', int(std * sr))

    return max_file_length, sr

def strip_silence(input_vector, threshold=35):
    split_vec = librosa.effects.split(input_vector, top_db = threshold)
    onset = split_vec[0][0]
    offset = split_vec[-1][-1]
    cut = input_vector[onset:offset]

    return cut

def preemphasis(input_vector, fs):
    '''
    2 simple high pass FIR filters in cascade to emphasize high frequencies
    and cut unwanted low-frequencies
    '''
    #first gentle high pass
    alpha=0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero,past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present,past)
    #second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.) # Normalize the frequency
    b, a = butter(8, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output

def onehot(value, range):
    '''
    int to one hot vector conversion
    '''
    one_hot = np.zeros(range)
    one_hot[value] = 1

    return one_hot

def pad_tensor_dims(predictors, time_dim, freq_dim):

    #zero-pad/cut time tim
    curr_time_dim = predictors.shape[2]
    curr_freq_dim = predictors.shape[3]

    if time_dim > curr_time_dim:
        #
        predictors_padded = np.zeros((predictors.shape[0],
                                                 predictors.shape[1],
                                                 time_dim,
                                                 predictors.shape[3]))
        predictors_padded[:,:,:curr_time_dim,:] = predictors
        predictors = predictors_padded

    elif time_dim < curr_time_dim:
        predictors = predictors[:,:,:time_dim,:]
    else:
        pass

    #zero-pad/cut freq tim
    if freq_dim > curr_freq_dim:
        #
        predictors_padded = np.zeros((predictors.shape[0],
                                                 predictors.shape[1],
                                                 predictors.shape[2],
                                                 freq_dim))
        predictors_padded[:,:,:,:curr_freq_dim] = predictors
        predictors = predictors_padded

    elif freq_dim < curr_freq_dim:
        predictors = predictors[:,:,:,:freq_dim]
    else:
        pass

    return predictors

def load_datasets(args):
    '''
    load preprocessed dataset dicts and output torch dataloaders
    '''

    print ('\n Loading dataset')
    loading_start = float(time.perf_counter())

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #PREDICTORS_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_predictors.npy')
    #TARGET_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_target.npy')
    PREDICTORS_LOAD = args.predictors_path
    TARGET_LOAD = args.target_path

    dummy = np.load(TARGET_LOAD,allow_pickle=True)
    dummy = dummy.item()
    #create list of datapoints for current fold
    foldable_list = list(dummy.keys())

    if args.shuffle_data:
        random.shuffle(foldable_list)

    fold_actors_list = folds_generator(args.num_folds, foldable_list, [args.train_perc, args.val_perc, args.test_perc])
    train_list = fold_actors_list[args.num_fold]['train']
    val_list = fold_actors_list[args.num_fold]['val']
    test_list = fold_actors_list[args.num_fold]['test']
    del dummy

    predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
    target_merged = np.load(TARGET_LOAD,allow_pickle=True)
    predictors_merged = predictors_merged.item()
    target_merged = target_merged.item()

    print ('\n building dataset for current fold')
    print ('\n training:')
    training_predictors, training_target = build_matrix_dataset(predictors_merged,
                                                                target_merged, train_list)
    print ('\n validation:')
    validation_predictors, validation_target = build_matrix_dataset(predictors_merged,
                                                                target_merged, val_list)
    print ('\n test:')
    test_predictors, test_target = build_matrix_dataset(predictors_merged,
                                                                target_merged, test_list)


    if args.reduce_training_set is not None:
        print ('Reduced training set: ', args.reduce_training_set)
        num_tr_data = training_predictors.shape[0]
        reduced_len = int(num_tr_data * args.reduce_training_set)
        training_predictors = training_predictors[:reduced_len]
        training_target = training_target[:reduced_len]
        validation_predictors = training_predictors[:reduced_len]
        validation_target = training_target[:reduced_len]

    if args.fast_test:
        print ('FAST TEST: using unly 100 datapoints ')
        #take only 100 datapoints, just for quick testing
        bound = args.fast_test_bound
        training_predictors = training_predictors[:bound]
        training_target = training_target[:bound]
        validation_predictors = validation_predictors[:bound]
        validation_target = validation_target[:bound]
        test_predictors = test_predictors[:bound]
        test_target = test_target[:bound]

    if args.predictors_normailzation == '01':
        print('normalize to 0 and 1')
        tr_max = np.max(training_predictors)
        #tr_max = 128
        training_predictors = np.divide(training_predictors, tr_max)
        validation_predictors = np.divide(validation_predictors, tr_max)
        test_predictors = np.divide(test_predictors, tr_max)

    elif args.predictors_normailzation == '0mean':
        print ('normalize to 0 mean and 1 std')
        tr_mean = np.mean(training_predictors)
        tr_std = np.std(training_predictors)
        training_predictors = np.subtract(training_predictors, tr_mean)
        training_predictors = np.divide(training_predictors, tr_std)
        validation_predictors = np.subtract(validation_predictors, tr_mean)
        validation_predictors = np.divide(validation_predictors, tr_std)
        test_predictors = np.subtract(test_predictors, tr_mean)
        test_predictors = np.divide(test_predictors, tr_std)
    else:
        raise ValueError('Invalid predictors_normailzation option')

    print ("Predictors range: ", np.min(training_predictors), np.max(training_predictors))

    #reshaping for cnn
    training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
    validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
    test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

    #cut/pad dims
    training_predictors = pad_tensor_dims(training_predictors, args.time_dim, args.freq_dim)
    validation_predictors = pad_tensor_dims(validation_predictors, args.time_dim, args.freq_dim)
    test_predictors = pad_tensor_dims(test_predictors, args.time_dim, args.freq_dim)

    print ('\nPadded dims:')
    print ('Training predictors: ', training_predictors.shape)
    print ('Validation predictors: ', validation_predictors.shape)
    print ('Test predictors: ', test_predictors.shape)

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float()
    val_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    train_target = torch.tensor(training_target).float()
    val_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()

    #build dataset from tensors
    tr_dataset = utils.TensorDataset(train_predictors, train_target)
    val_dataset = utils.TensorDataset(val_predictors, val_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)  #no batch here!!

    loading_time = float(time.perf_counter()) - float(loading_start)
    print ('\nLoading time: ' + str(np.round(float(loading_time), decimals=1)) + ' seconds')

    return tr_data, val_data, test_data
