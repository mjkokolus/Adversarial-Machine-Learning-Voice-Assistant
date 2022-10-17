# -*- Coding: utf-8 -*-
# @Time     : 2022/6/27 14:18
# @Author   : Linqi Xiao
# @Software : PyCharmlakers
# @Version  : python 3.6
# @Description :

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def decode_audio(audio_binary):
    audio, sample_rate = tf.audio.decode_wav(contents=audio_binary)  # audio: [samples, channels]
    return tf.squeeze(audio, axis=-1)


def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


def get_spectrogram(waveform):
    input_len = 16000
    if tf.shape(waveform) > input_len:
        waveform = waveform[: input_len]  # in this dataset, the max samples of waveform is 16000
    elif tf.shape(waveform) < input_len:
        zero_padding = tf.zeros(shape=[16000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)  # make sure waveform has the same type as zero padding
        waveform = tf.concat([waveform, zero_padding], axis=0)  # waveform with zero padding
    spectrogram = tf.signal.stft(signals=waveform, frame_length=255,
                                 frame_step=128)  # convert the waveform to a spectrogram via a STFT
    ## frame_length	An integer scalar Tensor. The window length in samples.
    ## frame_step	An integer scalar Tensor. The number of samples to step.
    spectrogram = tf.abs(spectrogram)  # magnitude + phase. use tf.abs to get rid of phase
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram, ax, debug=False):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        new_spectrogram = np.squeeze(spectrogram, axis=-1)  # (124, 129, 1) -> (124, 129)
    # Convert the frequencies to log scale and transpose, so that the time is represented on the x-axis (columns).
    log_spec = np.log(new_spectrogram.T + np.finfo(float).eps)
    if debug:
        print('old spectrogram shape: ', spectrogram.shape)
        print('new spectrogram shape: ', new_spectrogram.shape)
        print('log spec shape: ', log_spec.shape)
        print('spectrogram size: ', np.size(new_spectrogram))
    height = log_spec.shape[0]  # 129
    width = log_spec.shape[1]  # 124
    X = np.linspace(0, np.size(new_spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def predictModel(model, spectrogram):
    listOfData = [spectrogram]
    predictData = np.array(listOfData, dtype='float32')
    result = model.predict(predictData)
    lab = tf.argmax(result, 1)
    return lab


if __name__ == '__main__':
    wav_path = './audio_files/input.wav'
    my_waveform = get_waveform(wav_path)
    print(my_waveform.shape)
    my_spectrogram = get_spectrogram(my_waveform)
    print(my_spectrogram.shape)

    # fig, axes = plt.subplots(2, figsize=(12, 8))
    # timescale = np.arange(my_waveform.shape[0])
    # axes[0].plot(timescale, my_waveform.numpy())
    # axes[0].set_title('Waveform')
    # axes[0].set_xlim([0, 16000])
    #
    # plot_spectrogram(my_spectrogram.numpy(), axes[1], debug=True)
    # axes[1].set_title('Spectrogram')
    # plt.show()

    model = tf.saved_model.load(export_dir='F:\PythonProject\LearnTF\official_audio\Recognizing keywords\model')
    pred = predictModel(model, my_spectrogram)
    print(pred)
