#!/usr/bin/env python3

# %%
from operator import mod
from typing import List, Tuple

import librosa
import numpy as np
from hmmlearn import hmm
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import block
from scipy.sparse.construct import random
from sklearn.metrics import confusion_matrix
from jiwer import wer

import os

# be reproducible...
np.random.seed(1337)

# ---%<------------------------------------------------------------------------
# Part 1: Basics

# version 1.0.10 has 10 digits, spoken 50 times by 6 speakers
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nr = 50
speakers = list(['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'])
pathToSamples = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))) + '/res/free-spoken-digit-dataset-master/recordings'


# %%
def load_fts(digit: int, spk: str, n: int):
    # load sounds file, compute MFCC; eg. n_mfcc=13

    path_to_required_sample = os.path.join(pathToSamples, f"{str(digit)}_{spk}_{str(n)}.wav")
    y, sr = librosa.load(path_to_required_sample)
    return librosa.feature.mfcc(y, sr, None, n_mfcc=13).T
    pass

# %% Read mfccs

mfccs = {}
for speaker in speakers:
    print(speaker)
    digit_list = {}
    for digit in digits:
        digit_tries = []
        for i in range(0, nr):
            digit_tries.append(load_fts(digit, speaker, i))
        digit_list[digit] = digit_tries
    mfccs[speaker] = digit_list

# load data files and extract features

# %% 

# implement a 6-fold cross-validation (x/v) loop so that each speaker acts as
# test speaker while the others are used for training


def getXforDigit (digit: int, current_speakers: List[str]):
    X = []
    length = []
    for speaker in current_speakers:
        for try_ in mfccs[speaker][digit]:
            length.append(len(try_))
            X.append(try_)

    return np.concatenate(X, axis=0), length


def getModel():
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag",
                            init_params="cm", params="cmt")
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    model.transmat_ = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.5, 0.0, 0.0],
                                [0.0, 0.0, 0.5, 0.5, 0.0],
                                [0.0, 0.0, 0.0, 0.5, 0.5],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

    return model

def train_models(train_keys : List[str]):
    models = {}
    for digit in digits:
        model = getModel()
        X, length = getXforDigit(digit, train_keys)

        model.fit(X, length)
        models[digit] = model  
    
    return models


model_map = {}

def get_scores(speaker_pos : int):
        test_key = speakers[speaker_pos]
        train_keys = [x for x in speakers if test_key != x]
    
        models = train_models(train_keys)
        model_map[speakers[speaker_pos]] = models
        scores = evaluate(test_key, models)   
        return scores  
        
  
def evaluate(test_key : str, models : dict):
    scores = {}
    for i in digits:
        X_test, length_test = getXforDigit(i, [test_key])

        startpos = 0
        
        predictions_group = []
        for length in length_test:
            sample = X_test[startpos:startpos+length]
            startpos += length

            prediction_results = []
            for digit in digits:
                prediction_results.append(models[digit].score(sample))
            predictions_group.append(prediction_results)
        scores[i] = predictions_group
    return scores    
#%% Get Scores

results = {}
for pos, speaker in enumerate(speakers):
    results[speaker] = get_scores(pos)

#%%

matrices = []
for speaker in speakers:
    y_true = []
    y_pred = []
    for digit in digits: 
        samples = results[speaker][digit]
        for sample in samples:
            y_true.append(digit)
            y_pred.append(np.argmax(sample))
            

    matrices.append(confusion_matrix(y_true, y_pred))


#%% Plot

import matplotlib.pyplot as plt

for idx, cm in enumerate(matrices): 

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = digits
    plt.title(speakers[idx])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    for i in digits:
        for j in digits:
            plt.text(j,i, str(cm[i][j]))
    
    plt.savefig(f"../img/{speakers[idx]}_cm.png")
    plt.show()

   



# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
# note: you may find that one or more HMMs are performing particularly bad;
# what could be the reason and how to mitigate that?

# train the HMMs using the fit method; data needs to be concatenated,
# see https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439

# evaluate the trained models on the test speaker; how do you decide which word
# was spoken?

# compute and display the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# %%
# display the overall confusion matrix


# ---%<------------------------------------------------------------------------
# Part 2: Decoding

# %% 
def get_random_sequence(speaker: str) -> Tuple[List[int], List]:
    """
    Gets a random sequence of digits with a length between 3 and 6 of 
    the provided speaker.

    @return: the sequence of digits and corresponding mfcc-features 
    """
    sequence = []
    for _ in range(0, np.random.randint(3,7)):
        sequence.append(np.random.randint(0,10))
    
    mfcc_sequence = []
    for digit in sequence:
        mfcc_sequence.append(load_fts(digit, speaker, np.random.randint(0, nr)))

    return sequence, np.concatenate(mfcc_sequence, axis=0)

# generate test sequences; retain both digits (for later evaluation) and
# features (for actual decoding)

# combine the (previously trained) per-digit HMMs into one large meta HMM; make
# sure to change the transition probabilities to allow transitions from one
# digit to any other

def get_meta_hmm(speaker: str):
    """
    Gets a meta hmm which is trained on all speakers except the provided one
    """
    n_components = 5
    n_digits = 10
    transmat = np.zeros(shape=(n_digits*n_components,n_digits*n_components))
    means = []
    covars = []
    models = model_map[speaker]

    for idx in range(0,n_digits):
        
        tm = models[idx].transmat_
        means.append(models[idx].means_)
        covars.append(models[idx].covars_)

        start = idx*n_components
        end = start + n_components
        transmat[start:end,start:end] = tm

    for idx in range(0, nr):
        if (idx +1) % 5 == 0:
            transmat[idx] = np.concatenate([np.array([0.1, 0, 0 ,0, 0])] * 10)


    meta_model = hmm.GaussianHMM(n_components=nr, covariance_type="full")
    meta_model.startprob_ = np.concatenate([np.array([0.1, 0, 0 ,0, 0])] * 10) 
    meta_model.transmat_ = transmat
    meta_model.means_ = np.concatenate(means)
    meta_model.covars_ = np.concatenate(covars)
    return meta_model

# use the `decode` function to get the most likely state sequence for the test
# sequences; re-map that to a sequence of digits

def decode_meta(mffc_features, speaker: str):
    meta_hmm = get_meta_hmm(speaker)

    prob, sequence = meta_hmm.decode(mffc_features)

    states = []

    blocked = False
    for state in sequence:
        if state % 5 == 0 and not blocked:
            states.append(state // 5)
            blocked = True

        if state % 5 != 0:
            blocked = False
    return states

def list_to_str(digits: list):
    digit_str = [str(int) for int in digits]
    return " ".join(digit_str)

# use jiwer.wer to compute the word error rate between reference and decoded
# digit sequence

def compute_wer(speaker: str):
    digits_true, features_true = get_random_sequence(speaker)
    digits_pred = decode_meta(features_true, speaker)
    return wer(list_to_str(digits_true), list_to_str(digits_pred))

# compute overall WER (ie. over the cross-validation)

def compute_overall_wer(speaker : str = "theo"):
    digits_true, features_true = get_random_sequence(speaker)


    errors = []
    pred = []
    for speaker in speakers:
        digits_pred = decode_meta(features_true, speaker)
        pred.append(digits_pred)
        errors.append(wer(list_to_str(digits_true), list_to_str(digits_pred)))

    idx = np.argmin(errors)
    
    print(f"digits: {digits_true}")
    for i in range(0, len(speakers)):
        print(f" - {speakers[i]} predicted {pred[i]} with WER of {errors[i]}")

    return errors[idx], speakers[idx]

# ---%<------------------------------------------------------------------------
# Optional: Decoding

# %%
