from __future__ import print_function
import numpy as np
from scipy.io.wavfile import read, write
import sys
import librosa
#import essentia.standard as ess
#import essentia
import configparser
from scipy.signal import stft
import utility_functions as uf
import soundfile as sf
'''
Utility functions for audio-based data pre-processing
'''
cfg = configparser.ConfigParser()
cfg.read('preprocessing_config.ini')

#get values from config file
#global
SR = cfg.getint('sampling', 'sr_target')
FIXED_SEED = cfg.get('sampling', 'fixed_seed')
COMPRESSION = eval(cfg.get('feature_extraction', 'power_law_compression'))
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
NORMALIZATION = eval(cfg.get('feature_extraction', 'normalization'))
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SEQUENCE_LENGTH = cfg.getfloat('feature_extraction', 'sequence_length')
SEQUENCE_OVERLAP = cfg.getfloat('feature_extraction', 'sequence_overlap')
#spectrum
WINDOW_SIZE = cfg.getint('feature_extraction', 'window_size')
FFT_SIZE = cfg.getint('feature_extraction', 'fft_size')
HOP_SIZE_STFT = cfg.getint('feature_extraction', 'hop_size_stft')
WINDOW_TYPE = str(cfg.get('feature_extraction', 'window_type'))
#cqt
HOP_SIZE_CQT = cfg.getint('feature_extraction', 'hop_size_cqt')
BINS_PER_OCTAVE = cfg.getint('feature_extraction', 'bins_per_octave')
N_BINS = cfg.getint('feature_extraction', 'n_bins')
FMIN = cfg.getint('feature_extraction', 'fmin')
#mfcc
WINDOW_SIZE_MFCC = cfg.getint('feature_extraction', 'window_size_mfcc')
FFT_SIZE_MFCC = cfg.getint('feature_extraction', 'fft_size_mfcc')
HOP_SIZE_MFCC = cfg.getint('feature_extraction', 'hop_size_mfcc')
WINDOW_TYPE_MFCC = str(cfg.get('feature_extraction', 'window_type_mfcc'))
N_MFCC = cfg.getint('feature_extraction', 'n_mfcc')
#melspectrogram
HOP_SIZE_MEL = cfg.getint('feature_extraction', 'hop_size_mel')
FFT_SIZE_MEL = cfg.getint('feature_extraction', 'fft_size_mel')

FIXED_SEED = eval(FIXED_SEED)

if FIXED_SEED is not None:
    # Set seed
    manualSeed = FIXED_SEED
    seed=manualSeed
    np.random.seed(seed)


if AUGMENTATION:
    import augmentation

def spectrum_fast(x):
    f, t, seg_stft = stft(x,window='hamming',nperseg=256,noverlap=128)

    return np.rot90(np.abs(seg_stft))


def spectrum(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE_STFT, fs=SR, window_type=WINDOW_TYPE, compression=COMPRESSION):
    '''
    magnitudes spectrum
    '''
    SP = librosa.core.stft(x, n_fft=N, hop_length=H, window=window_type)
    SP = np.abs(SP)
    if compression:
        SP = np.power(SP, 2./3.)  #power law compression
    SP = np.rot90(SP)

    return SP


def spectrum_CQ(x, H=HOP_SIZE_CQT, fs=SR, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS, fmin=FMIN, compression=COMPRESSION):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    CQT = librosa.core.cqt(x, hop_length=H, sr=fs, bins_per_octave=24, n_bins=168, fmin=55)
    CQT = np.abs(CQT)
    if compression:
        CQT = np.power(CQT, 2./3.)  #power law compression
    CQT = np.rot90(CQT)

    return CQT

def spectrum_mel(x, H=HOP_SIZE_MEL, fs=SR, N=FFT_SIZE_MEL, compression=COMPRESSION):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    MEL = librosa.feature.melspectrogram(x, sr=fs, n_fft=N, hop_length=H)
    MEL = np.abs(MEL)
    if compression:
        MEL = np.power(MEL, 2./3.)  #power law compression
    MEL = np.rot90(MEL)

    return MEL

'''
def mfcc(x, M=WINDOW_SIZE_MFCC, N=FFT_SIZE_MFCC, H=HOP_SIZE_MFCC, fs=SR,
            window_type=WINDOW_TYPE_MFCC, n_mfcc=N_MFCC):

	#-extract features from audio file
	#-Features:
	#	MFCC (24 COEFFS)


	#audioLoader = ess.EasyLoader(filename=file_name, sampleRate=fs)
	#create essentia instances
	x = essentia.array(x)
	spectrum = ess.Spectrum(size=N)
	window = ess.Windowing(size=M, type=window_type)
	mfcc = ess.MFCC(numberCoefficients=n_mfcc, inputSize=int(N/2+1), sampleRate=fs, highFrequencyBound=int(fs/2-1))

	#init vectors
	MFCC = []


	#compute features for every stft frame
	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
		wX = window(frame)  #window frame
		mX = spectrum(wX)  #compute fft

		mfcc_bands, mfcc_coeffs = mfcc(mX)
		MFCC.append(mfcc_coeffs)


	#convert into numpy matrices
	MFCC = essentia.array(MFCC)

	return MFCC
'''

def extract_features(input_vector, features_type):
    if features_type == 'stft':
        feats = spectrum(input_vector)
    elif features_type == 'cqt':
        feats = spectrum_CQ(input_vector)
    elif features_type == 'mel':
        feats = spectrum_mel(input_vector)
    elif features_type == 'mfcc':
        feats = mfcc(input_vector)
    elif features_type == 'spectrum_fast':
        feats = spectrum_fast(input_vector)
    else:
        raise ValueError('Wrong features_type. Possible values: stft, cqt, mfcc')

    return feats


def preprocess_datapoint(input_vector, max_file_length):
    '''
    generate predictors (stft) and target (valence sequence)
    of one sound file from the OMG dataset
    '''
    if SEGMENTATION:

        seq_len_samps = int(SEQUENCE_LENGTH * SR)
        # if segment cut initial and final silence if present
        #samples = uf.strip_silence(raw_samples)
        if len(input_vector) < seq_len_samps:
            pad = np.zeros(seq_len_samps)
            pad[:len(input_vector)] = input_vector
            input_vector = pad

    else:
        #if not, zero pad all sounds to the same length
        pad = np.zeros(max_file_length)
        if len(input_vector) > max_file_length:
            pad[:max_file_length-1] = input_vector[:max_file_length-1]
        else:
            pad[:len(input_vector)] = input_vector  #zero padding
        input_vector = pad
    #print ("COGLIONEEEEEEEEEEE, ", input_vector.shape)
    feats = extract_features(input_vector, FEATURES_TYPE)  #extract features

    return feats

def segment_datapoint(features, label):
    '''
    segment features of one long features vector
    into smaller matrices of length "sequence_length"
    and overlapped by "sequence_overlap"
    This function applies the same label to every segmented datapoint!!
    -- label_function is the function that extracts the label
    '''
    #compute how many frames per sequence
    seq_len_samps = int(SEQUENCE_LENGTH * SR)
    dummy_samps = np.zeros(seq_len_samps)
    dummy_feats = feats = extract_features(dummy_samps, FEATURES_TYPE)
    seq_len_frames = dummy_feats.shape[0]
    num_frames = features.shape[0]

    #create pointer for segmentation
    step = int(np.round(seq_len_frames*SEQUENCE_OVERLAP))  #segmentation overlap step
    pointer = np.arange(0, num_frames, step, dtype='int')  #initail positions of segments

    #init vectors
    predictors = []
    target = []

    #segment arrays and append datapoints to vectors
    if SEGMENTATION:
        for start in pointer:
            stop = int(start + seq_len_frames)
            #print (start, stop, num_frames)
            if stop <= num_frames:
                temp_predictors = features[start:stop]
                predictors.append(temp_predictors)
                target.append(label)
            else:  #last datapoint has a different overlap
                #compute last datapoint only i vector is enough big
                if num_frames > seq_len_frames + (seq_len_frames*SEQUENCE_OVERLAP):
                    temp_predictors = features[-int(seq_len_frames):]
                    predictors.append(temp_predictors)
                    target.append(label)
                    pass
    else:
        predictors.append(features)
        target.append(label)
    predictors = np.array(predictors)
    target = np.array(target)

    return predictors, target

def preprocess_foldable_item(sounds_list, max_file_length, get_label_function, print_item_progress=False):
    '''
    compute predictors and target of all sounds in sound list
    sound_list should contain all filenames of 1 single foldable item
    '''

    predictors = []
    target = []
    #print(len(sounds_list))

    #librosa sr is None if no resampling is required (speed up)

    if len(sounds_list) > 1:
        try:
            sr, dummy = read(sounds_list[0])
            if sr == SR:
                librosa_SR = None
            else:
                librosa_SR = SR
        except:
            librosa_SR = SR
    else:
        librosa_SR = SR

    librosa_SR = SR

    #process all files in sound_list
    index_ = 0
    for sound_file in sounds_list:
        try:
            label = get_label_function(sound_file)
            #print (sound_file)
            samples, sr = librosa.core.load(sound_file, sr=librosa_SR)  #read audio
            print (len(samples))
            print (np.max(samples))
            if np.max(samples) > 0.005:  #if sound is not empty
                if NORMALIZATION:
                    samples = np.divide(samples, np.max(samples))
                    samples = np.multiply(samples, 0.8)
                if AUGMENTATION:
                    curr_list = [samples]
                    for i in range(NUM_AUG_SAMPLES):
                        temp_aug = augmentation.gen_datapoint(samples)
                        curr_list.append(temp_aug)

                else:
                    curr_list = [samples]

                for sound in curr_list:

                        long_predictors = preprocess_datapoint(sound, max_file_length)  #compute features
                        cut_predictors, cut_target = segment_datapoint(long_predictors, label)   #segment feature maps
                        if not np.isnan(np.std(cut_predictors)):   #some sounds give nan for no reason
                            for i in range(cut_predictors.shape[0]):
                                predictors.append(cut_predictors[i])
                                target.append(cut_target[i])

                            #print ('Foldable item progress:')
            else:
                print ('silent sample')

        except Exception as e:
            print ('\r corrupted file found: not added to dataset')
            #print (e)
            #raise ValueError(e)
            #pass



        index_ += 1
        if print_item_progress:
            uf.print_bar(index_, len(sounds_list))

    predictors = np.array(predictors)
    target = np.array(target)

    return predictors, target

def shuffle_datasets(predictors, target):
    '''
    random shuffle predictors and target matrices
    '''
    shuffled_predictors = []
    shuffled_target = []
    num_datapoints = target.shape[0]
    random_indices = np.arange(num_datapoints)
    np.random.shuffle(random_indices)
    for i in random_indices:
        shuffled_predictors.append(predictors[i])
        shuffled_target.append(target[i])
    shuffled_predictors = np.array(shuffled_predictors)
    shuffled_target = np.array(shuffled_target)

    return shuffled_predictors, shuffled_target
