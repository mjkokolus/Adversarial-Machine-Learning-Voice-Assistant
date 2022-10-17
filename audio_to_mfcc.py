# -*- Coding: utf-8 -*-
# @Time     : 2022/6/29 16:22
# @Author   : Linqi Xiao
# @Software : PyCharm
# @Version  : python 3.6
# @Description :

import tensorflow as tf


def audio_to_mfcc(audio_contents, channels=1, sample_rate=16000, dct_counts=20, debug=False):
    waveform, sample_rate = tf.audio.decode_wav(contents=audio_contents,
                                                desired_channels=channels,
                                                desired_samples=sample_rate)
    waveform = tf.squeeze(waveform, axis=-1)
    input_len = 16000
    if tf.shape(waveform) > input_len:
        waveform = waveform[: input_len]  # in this dataset, the max samples of waveform is 16000
    elif tf.shape(waveform) < input_len:
        zero_padding = tf.zeros(shape=[16000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)  # make sure waveform has the same type as zero padding
        waveform = tf.concat([waveform, zero_padding], axis=0)  # waveform with zero padding
    stft = tf.signal.stft(signals=waveform, frame_length=640, frame_step=640, fft_length=1024)
    spectrogram = tf.abs(stft)
    spectrogram = tf.square(spectrogram)

    # Warp the linear scale spectrogram into the mel-scale
    num_spectrogram_bins = stft.shape.as_list()[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80, tf.cast(x=sample_rate / 2, dtype=tf.float32), 128
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,
                                                                        num_spectrogram_bins=num_spectrogram_bins,
                                                                        sample_rate=sample_rate,
                                                                        lower_edge_hertz=lower_edge_hertz,
                                                                        upper_edge_hertz=upper_edge_hertz)
    spectrogram = tf.sqrt(spectrogram)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs from log_mel_spectrogram
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    if debug:
        print('waveform shape:            ', waveform.shape)
        print('stft shape:                ', stft.shape)
        print('spectrogram shape:         ', spectrogram.shape)
        print('mel_spectrogram shape:     ', mel_spectrogram.shape)
        print('log_mel_spectrogram shape: ', log_mel_spectrogram.shape)
        print('mfcc shape:                ', mfcc.shape)
        print('sample_rate:               ', sample_rate)
        print('num_spectrogram_bins:      ', num_spectrogram_bins)
        print('num_mel_bins:              ', num_mel_bins)
        print('lower_edge_hertz:          ', lower_edge_hertz)
        print('upper_edge_hertz:          ', upper_edge_hertz)

    mfcc = mfcc[..., :dct_counts]
    return mfcc


if __name__ == '__main__':
    input_file = tf.constant('F:\PythonProject/voice/audio_files\input.wav')

    # Compute the mfcc
    audio = tf.io.read_file(input_file)
    mfcc = audio_to_mfcc(audio, debug=True)

    # Get only the first 20
    print(mfcc.shape)
