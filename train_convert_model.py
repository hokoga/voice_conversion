#coding:utf-8

import numpy as np
import random
import sys

from stf import STF
from mfcc import MFCC
from arbm import ARBM


def calc_mfcc(filename, mfcc_dim):
    stf = STF()
    stf.loadfile(filename)
    mfcc_vals = np.empty((mfcc_dim, 0), "float")

    mfcc = MFCC(stf.SPEC.shape[1] * 2, stf.frequency, dimension = mfcc_dim, channels = mfcc_dim)
    for j in range(stf.SPEC.shape[0]):
        res = mfcc.mfcc(stf.SPEC[j])
        mfcc_vals = np.hstack([mfcc_vals, res.reshape(mfcc_dim, 1)])

    return mfcc_vals


def shuffle_array(arr):
    for i in range(arr.shape[1]):
        a = np.random.randint(arr.shape[1])
        b = np.random.randint(arr.shape[1])

        arr[:,a], arr[:,b] = np.array(arr[:,b]), np.array(arr[:,a])


def normalize(arr):
    normed_arr = (arr.T - np.average(arr, axis = 1)) / np.std(arr, axis = 1)
    return normed_arr.T


def restore(arr, ave, std):
    restored_arr = arr.T * std + ave
    return restored_arr.T


def train(model, mfcc_vals, batch_size, rate, s):
    shuffle_array(mfcc_vals)
    for j in range(int(np.ceil((mfcc_vals.shape[1] / float(batch_size))))):
        start = j * batch_size
        end = (j + 1) * batch_size
        model.train(rate, mfcc_vals[:, start:end].reshape(model.visible_size , 1, -1), s)
                

    return model

def train_w_o_weight(model, mfcc_vals, batch_size, rate, s):
    shuffle_array(mfcc_vals)
    for j in range((mfcc_vals.shape[1] / batch_size)):
        start = j * batch_size
        end = (j + 1) * batch_size
        model.train_w_o_weight(rate, mfcc_vals[:, start:end].reshape(model.visible_size, 1, -1), s)

    return model


def calc_error(model, mfcc, s):
    mfcc_size = mfcc.shape[0]
    length = mfcc.shape[1]
    error = np.zeros([mfcc_size, 1])

    for i in range(length):
        input_mfcc = mfcc[:,i].reshape(mfcc_size, 1)
        error += np.square(input_mfcc - model.construct(input_mfcc, s, s))

    return np.sum(error / length)


def get_mfcc_list(train_speaker_list, data_num):
    mfcc_list = []
    for s, speaker in enumerate(train_speaker_list):
        mfcc_vals = []
        for i in range(1, data_num + 1):
            filename = "shirobako/" + speaker + "/" + speaker + str(i) + ".stf"
            mfcc_vals.append(calc_mfcc(filename, model.visible_size))
        mfcc_list.append(mfcc_vals)

    return mfcc_list


def train_first_step(model, speaker_num, data_num, mfcc_list, 
        repeat_num, batch_size, rate):
    for k in range(repeat_num):
        for i in range(data_num):
            for j in range(speaker_num):
                train(model, mfcc_list[j][i], batch_size, rate, j)

        #calc error
        err = 0.0
        for j in range(speaker_num):
            err += calc_error(model, mfcc_list[j][0], j)
        err /= speaker_num
        print "k = " + str(k) + ": " + str(err)


def train_second_step(model, train_speaker_num, speaker_num, data_num,
        mfcc_list, repeat_num, batch_size, rate):
    for k in range(repeat_num):
        for i in range(data_num):
            for j in range(speaker_num):
                train_w_o_weight(model, mfcc_list[j][i], batch_size, rate,
                        j + train_speaker_num)

        #calc error
        err = 0.0
        for j in range(speaker_num):
            err += calc_error(model, mfcc_list[j][0], j + train_speaker_num)
        err /= speaker_num
        print "k = " + str(k) + ": " + str(err)


if __name__ == "__main__":
    mfcc_dim = 32 
    data_num = 10
    hidden_size = 256
    batch_size = 200
    repeat_num = 5
    rate = 0.0003
    train_speaker_list = ["yano", "aoi", "midori", "misa"]

    model = ARBM()
    model.initialize(mfcc_dim, hidden_size, len(train_speaker_list) + 2)

    print "Calculating mfcc..."
    mfcc_list = get_mfcc_list(train_speaker_list, data_num)

    #Normalize mfcc
    for i in range(len(train_speaker_list)):
        for j in range(data_num):
            mfcc_list[i][j] = normalize(mfcc_list[i][j])
            
    print "Train step1 start"
    train_first_step(model, len(train_speaker_list), data_num, mfcc_list, 
            repeat_num, batch_size, rate)


    print "Calculating mfcc..."
    speaker_list = ["honda", "shizuka"]
    data_num = 5
    mfcc_list = get_mfcc_list(speaker_list, data_num)

    #Normalize mfcc
    for i in range(len(speaker_list)):
        for j in range(data_num):
            mfcc_list[i][j] = normalize(mfcc_list[i][j])


    print "Train step2 start"
    rate = 0.00005
    repeat_num = 20
    train_second_step(model, len(train_speaker_list), len(speaker_list), 
            data_num, mfcc_list, repeat_num, batch_size, rate)
            

    model.save("shirobako_param_h_" + str(hidden_size))
