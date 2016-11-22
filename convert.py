#coding:utf-8

import numpy as np
import copy

from stf import STF
from mfcc import MFCC
from train_convert_model import *
from arbm import ARBM


def smoothing(mfcc_vals, size):
    mfcc = copy.deepcopy(mfcc_vals)
    for i in range(mfcc_vals.shape[1]):
        if i <= size - 1:
            val = mfcc_vals[:,i]
            for s in range(1, 2 * size + 1):
                val += mfcc[:,i+s]
            mfcc_vals[:,i] = val / (2 * size + 1)
        elif i >= mfcc_vals.shape[1] - size:
            val = mfcc_vals[:,i]
            for s in range(1, 2 * size + 1):
                val += mfcc[:,i-s]
            mfcc_vals[:,i] = val / (2 * size + 1)
        else:
            val = mfcc_vals[:,i]
            for s in range(1, size + 1):
                val += mfcc[:,i-s] + mfcc[:,i+s]
            mfcc_vals[:,i] = val / (2 * size + 1)


def get_pitch_avg_and_std(speaker, data_num):
    pitch = np.empty(0)
    stf = STF()
    for i in range(1, data_num + 1):
        filename = "shirobako/" + speaker + "/" + speaker + str(i) + ".stf"
        stf.loadfile(filename)
        pitch = np.hstack([pitch, np.delete(stf.F0, np.where(stf.F0 == 0.0))])

    avg = np.average(pitch)
    std = np.std(pitch)

    return (avg, std)


def convert_pitch(source_file, source_avg, source_std, target_avg, target_std):
    stf = STF()
    stf.loadfile(source_file)
    pitch = stf.F0

    pitch[np.where(pitch != 0)] = (target_std / source_std) * (pitch[np.where(pitch != 0)] - source_avg) + target_avg

    return pitch


def save_spec(source_file, output_file, mfcc_vals, mfcc_dim, pitch):
    stf = STF()
    stf.loadfile(source_file)

    mfcc = MFCC(stf.SPEC.shape[1] * 2, stf.frequency, dimension = mfcc_dim, channels = mfcc_dim)
    for i in range(mfcc_vals.shape[1]):
        stf.SPEC[i] = mfcc.imfcc(mfcc_vals[:,i])
    stf.F0 = pitch

    stf.savefile(output_file)


if __name__ == "__main__":
    hidden_size = 256
    filename = "shirobako_param_h_" + str(hidden_size)

    model = ARBM()
    model.load(filename)
    speaker_num = {}
    speaker_num["honda"] = 4
    speaker_num["shizuka"] = 5
    data_num = 5


    #calc mfcc of source sound
    source_file = "shirobako/honda/bansaku1.stf"
    mfcc_vals = calc_mfcc(source_file, model.visible_size)

    #calc average and stf of mfcc
    ave = np.average(mfcc_vals, axis=1)
    std = np.std(mfcc_vals, axis=1)

    #normalize
    mfcc_vals = normalize(mfcc_vals)

    #convert mfcc
    out = np.zeros([model.visible_size, mfcc_vals.shape[1]])
    for i in range(mfcc_vals.shape[1]):
        input_mfcc = mfcc_vals[:,i].reshape(model.visible_size, 1)
        out[:,i] = model.construct(input_mfcc, speaker_num["honda"], 
                speaker_num["shizuka"]).reshape(model.visible_size)
    restored_out = restore(out, ave, std)
    smoothing(restored_out, 2)

    #convert pitch
    source_avg, source_std = get_pitch_avg_and_std("honda", data_num)
    target_avg, target_std = get_pitch_avg_and_std("shizuka", data_num)
    converted_pitch = convert_pitch(source_file, source_avg, source_std, 
            target_avg, target_std)

    #save mfcc and pitch to stf file
    save_spec(source_file, "bansaku_converted.stf", 
            restored_out, model.visible_size, converted_pitch)
