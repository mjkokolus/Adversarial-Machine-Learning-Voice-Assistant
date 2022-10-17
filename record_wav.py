# -*- Coding: utf-8 -*-
# @Time     : 2022/6/26 19:08
# @Author   : Linqi Xiao
# @Software : PyCharm
# @Version  : python 3.6
# @Description :


import os
import wave
import time
import pyaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_io as tfio


def load_wav_16k_mono(filename, debug=False):
    file_contents = tf.io.read_file(filename=filename)
    wav, sample_rate = tf.audio.decode_wav(contents=file_contents, desired_channels=1)
    new_wav = tf.squeeze(wav, axis=-1)
    new_sample_rate = tf.cast(sample_rate, dtype=tf.int64)  # int32 -> int64
    new_wav = tfio.audio.resample(input=new_wav, rate_in=new_sample_rate, rate_out=16000)
    if debug:
        print('* wav shape:    ', wav.shape)
        # print(sample_rate)
        print('* new_wav shape:', new_wav.shape)
    return new_wav


class recoder:
    def __init__(self, chunk=1024, audio_channel=1, audio_rate=16000, record_seconds=2, input_filename='input.wav'):
        self.chunk = chunk
        self.audio_format = pyaudio.paInt16
        self.audio_channel = audio_channel
        self.audio_rate = audio_rate
        self.record_seconds = record_seconds
        self.input_filename = input_filename
        self.input_filepath = './audio_files'
        self.path = os.path.join(self.input_filepath, self.input_filename)
        print('* path --> ', self.path)

    def record(self):
        p = pyaudio.PyAudio()
        print('>> Start recording')
        for i in range(1, 4):
            time.sleep(1)
            print(i)
        stream = p.open(rate=self.audio_rate, channels=self.audio_channel, format=self.audio_format, input=True,
                        frames_per_buffer=self.chunk)
        frames = []
        for _ in tqdm(range(0, int(self.audio_rate / self.chunk * self.record_seconds))):
            data = stream.read(self.chunk)
            frames.append(data)
        print('>> Stop recording')
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('... All stream closed')
        with wave.open(self.path, 'wb') as wf:
            wf.setnchannels(self.audio_channel)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.audio_rate)
            wf.writeframes(b''.join(frames))
        print('... Finish saving')

    # def play(self):
    #     wf = wave.open(self.path, 'rb')
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
    #                     rate=wf.getframerate(), output=True)
    #     data = wf.readframes(self.chunk)
    #     while data != b'':
    #         stream.write(data)
    #         data = wf.readframes(self.chunk)
    #     stream.stop_stream()
    #     stream.close()
    #     p.terminate()
    #     print('... All stream closed')

    def draw(self):
        testing_wav_data = load_wav_16k_mono(self.path, debug=True)
        plt.plot(testing_wav_data)
        plt.show()


if __name__ == '__main__':
    recoder = recoder()
    recoder.record()
    recoder.draw()

