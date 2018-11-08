#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import wave
import math
import random
import argparse


DEFAULT_FACTOR = 0.045
DEFAULT_RATE = 44100


def img_to_spec(image_name, sound_name, invert=False, limit_amp=True, show_prog=True, duration=None, fs=DEFAULT_RATE, fmax=None, factor=DEFAULT_FACTOR):
    img_data = load_image(image_name, invert=invert)
    if duration is not None:
        t_num = duration * fs
        if fmax is None:
            img_data = resize(img_data, t_num)
        else:
            freq_ratio = fmax / fs
            img_data = resize(img_data, t_num, freq_ratio=freq_ratio)
    img_data = convert_image(img_data, factor)

    if fmax is None:
        sig_data = generate_sound(img_data, show_prog=show_prog)
    else:
        freq_ratio = fmax / fs
        sig_data = generate_sound(img_data, freq_ratio=freq_ratio, show_prog=show_prog)

    if limit_amp:
        UINT8_MAX = 255
        amp_ratio = img_data.max() / math.exp(factor * UINT8_MAX)
        save_sound(sig_data, sound_name, fs=fs, amp_ratio=amp_ratio)
    else:
        save_sound(sig_data, sound_name, fs=fs)


def load_image(fname, invert=False):
    img = Image.open(fname)
    img = img.convert('L')
    img = ImageOps.flip(img)
    if invert:
        img = ImageOps.invert(img)
    return np.asarray(img).T


def resize(img_data, t_num, freq_ratio=(1 / 2)):
    x_num, y_num = img_data.shape
    m = (-y_num + math.sqrt(y_num ** 2 + 4 * x_num * y_num * t_num * freq_ratio)) / (2 * x_num * y_num)
    x_num2, y_num2 = round(m * x_num), round(m * y_num)

    img = Image.fromarray(img_data.T)
    img = img.resize((x_num2, y_num2), resample=Image.BILINEAR)
    return np.asarray(img).T


def convert_image(img_data, factor):
    img_data2 = np.zeros(img_data.shape)
    nz = (img_data != 0)
    img_data2[nz] = np.exp(factor * img_data[nz])
    return img_data2


def generate_sound(img_data, freq_ratio=(1 / 2), show_prog=False):
    x_num, y_num = img_data.shape
    f_num = math.ceil((1 / freq_ratio) * y_num)
    w_len = 2 * f_num
    t_num = (x_num - 1) * f_num + w_len
    w_num = 2 * t_num - w_len
    window = window_function(np.arange(-w_num // 2, w_num // 2), f_num)
    t = np.arange(f_num)
    sgn = lambda x: 1 if x >= 0 else -1
    it = tqdm if show_prog else iter

    sig_data = np.zeros(t_num)
    w_sum = np.zeros(t_num)
    for x in it(range(x_num)):
        s = np.zeros(f_num)
        t0 = w_num - t_num - x * f_num
        r = random.choice(np.arange(y_num)[img_data[x] != 0]) if (img_data[x] != 0).any() else 0
        for y in range(y_num):
            y = (y + r) % y_num
            p = np.abs(s).argmax()
            s += -sgn(s[p]) * img_data[x, y] * np.cos(((2 * math.pi / f_num) * y) * (t - p))
        for i in range(0, t_num, f_num):
            sig_data[i:i + f_num] += window[i + t0:i + t0 + f_num] * s
        w_sum += window[t0:t0 + t_num]

    return sig_data / w_sum


def save_sound(sig_data, fname, fs=DEFAULT_RATE, amp_ratio=1):
    sig_data = convert_sound(sig_data, amp_ratio=amp_ratio)

    with wave.open(fname, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(sig_data)


def convert_sound(sig_data, amp_ratio=1):
    INT16_MIN, INT16_MAX, INT16_NUM = -32768, 32767, 65536
    sig_data = (amp_ratio * (INT16_NUM / 2) / np.abs(sig_data).max()) * sig_data
    sig_data = np.floor(sig_data)
    sig_data = sig_data.clip(INT16_MIN, INT16_MAX)
    sig_data = sig_data.astype(np.int16)
    return sig_data


def window_function(t, f_num):
    return np.exp(-math.pi * (t / f_num) ** 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image to spectrogram.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('image_name', metavar='imagefile', help='input image file')
    parser.add_argument('sound_name', metavar='soundfile', help='output sound file (.wav)')
    parser.add_argument('-i', '--invert', action='store_true', help='invert the amplitude of spectrogram')
    parser.add_argument('-a', '--amplify', action='store_true', help='amplify sound as much as possible')
    parser.add_argument('-q', '--quiet', action='store_true', help='do not print the progress bar')
    parser.add_argument('-l', '--length', dest='duration', metavar='LEN', type=float, help='length of sound [s]')
    parser.add_argument('-s', '--sampling-frequency', dest='fs', metavar='FS', type=float, default=DEFAULT_RATE, help='sampling frequency of sound [Hz]')
    parser.add_argument('-m', '--maximum-frequency', dest='fmax', metavar='FMAX', type=float, help='maximum frequency of sound [Hz]')
    parser.add_argument('-f', '--factor', metavar='FACTOR', type=float, default=DEFAULT_FACTOR, help='amplitude factor')

    args = parser.parse_args()
    limit_amp = not args.amplify
    show_prog = not args.quiet
    img_to_spec(args.image_name, args.sound_name, invert=args.invert, limit_amp=limit_amp, show_prog=show_prog, duration=args.duration, fs=args.fs, fmax=args.fmax, factor=args.factor)
