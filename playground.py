# @Author: shounak
# @Date:   2022-05-14T14:55:56-07:00
# @Email:  shounak@stanford.edu
# @Filename: playground.py
# @Last modified by:   shounak
# @Last modified time: 2022-05-20T01:12:31-07:00

import collections
import numpy as np
from math import log2, pow
import subprocess
import librosa.display
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

_ = """########################################################################
############################### INSTALLATION NOTES ############################
############################################################################"""
"""
> librosa has numba as a dependency. To use librosa, we must install the arm64
> version of numba. To do this, we ran the command:
> arch -arm64 pip install numba --no-binary :all:
>                 and then
> arch -arm64 pip install librosa --no-binary :all:

> Command run for pyaudio (as per https://stackoverflow.com/questions/68251169
                                  /unable-to-install-pyaudio-on-m1-mac-portaudio
                                  -already-installed):
> brew install portaudio
> python3 -m pip install --global-option='build_ext' \
>        --global-option='-I/opt/homebrew/Cellar/portaudio/19.7.0/include' \
>        --global-option='-L/opt/homebrew/Cellar/portaudio/19.7.0/lib' pyaudio

> pip3 install PyObjC

"""

_ = """########################################################################
#################################### PLANNING #################################
############################################################################"""
"""
Objective: Go from audio/video recording to:
– Cool and meaningful visualization of music
– Split different voices based on conditional probabilities (bayes thm.)
– Visualize accordingly
– Transcribe into sheet music (once delineation is complete)
– For differentation, include live-streaming version.
"""

_ = """########################################################################
################################ HYPERPARAMETERS ##############################
############################################################################"""

CODEC_MAPPING = {'mp4': 'libx264',
                 'ogv': 'libtheora',
                 'webm': 'libvpx',
                 'ogg': 'libvorbis',
                 'mp3': 'pcm_s16le',
                 'wav': 'libvorbis',
                 'm4a': 'libfdk_aac'}
PIANO_RANGE = (27, 4186)

_ = """########################################################################
################################## DEFINTIONS #################################
############################################################################"""

# def isfile(fname):
#     return os.path.isfile(fname)


def get_note(freq):
    """https://www.johndcook.com/blog/2016/02/10/musical-pitch-notation/."""
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    h = round(12 * log2(freq / C0))
    octave = h // 12
    n = h % 12
    return NAME[n] + str(octave)


def convert_file(input_path, output_path):
    if input_path == output_path:
        raise ValueError('You entered identical paths.')
    subprocess.call(['ffmpeg', '-i', input_path,
                     output_path])
    return output_path


def plot_waveform(waveform):
    plt.figure(figsize=(20, 8))
    plt.plot(waveform)


def librosa_plot_spectrogram(waveform):
    """ LIBROSA METHOD """
    STFT_waveform = librosa.stft(waveform)  # STFT of y
    STFT_DB_waveform = librosa.amplitude_to_db(
        np.abs(STFT_waveform), ref=np.max)
    fig, ax = plt.subplots(figsize=(18, 15))
    img = librosa.display.specshow(STFT_DB_waveform,
                                   x_axis='time',
                                   y_axis='linear',
                                   ax=ax)
    ax.set(title='Spectrogram for audio file')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return STFT_DB_waveform


def util_fill_in_gaps(notes_captured):
    prev_index, prev_note = list(notes_captured.items())[0]
    for next_index, next_note in list(notes_captured.items())[1:]:
        if next_index == prev_index + 2:
            notes_captured[prev_index + 1] = f'{prev_note}-{next_note}'
        elif next_index > prev_index + 2:
            print('Ummmm...huge gap in mapping.')
        prev_index = next_index
        prev_note = next_note
    # Remember, mapping is reversed!!
    return collections.OrderedDict(sorted(notes_captured.items(), reverse=True))


def librosa_plot_chroma(waveform, sampling_rate):
    chroma = librosa.feature.chroma_cqt(y=waveform, sr=sampling_rate)
    fig, ax = plt.subplots(figsize=(18, 15))
    img = librosa.display.specshow(chroma,
                                   y_axis='chroma', x_axis='time',
                                   ax=ax)
    ax.set(title='Chromagram demonstration')
    fig.colorbar(img, ax=ax)

    print(ax.get_yticklabels())
    extracted_data = chroma  # img.get_array().reshape(chroma.shape)
    notes_captured = {i.get_position()[1]: i.get_text()
                      for i in ax.get_yticklabels()
                      if i.get_position()[1] < len(extracted_data)}

    cont_notes_captured = util_fill_in_gaps(notes_captured)
    # chroma_labeled = {}
    # for data_pos in sorted(range(0, len(extracted_data)), reverse=True):
    #     chroma_labeled[cont_notes_captured.get(
    #         data_pos)] = extracted_data[data_pos]

    return extracted_data, cont_notes_captured


def plot_spectrogram(waveform, sampling_rate):
    fig, ax = plt.subplots(figsize=(18, 15))
    cmap = plt.get_cmap('viridis')
    spectrum, row_freqs, col_mpoints, cax = plt.specgram(
        waveform, Fs=sampling_rate, cmap=cmap, mode='magnitude', NFFT=256,
        scale='dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    cbar = fig.colorbar(cax)
    cbar.set_label('Amplitude', rotation=270)
    return spectrum, row_freqs, col_mpoints


def plot_modified_spectrogram(spectrum, PIANO_RANGE=PIANO_RANGE, masked=True,
                              skip_modify=False, tuning=0.1):
    spectrum = np.flipud(spectrum)
    _ = plt.figure(figsize=(18, 15))
    if not skip_modify:
        spectrum = spectrum ** tuning
        if masked:
            spectrum = spectrum[PIANO_RANGE[0]:PIANO_RANGE[1], :]
    _ = sns.heatmap(spectrum)
    return spectrum


def plot_entire_freq_hist(spectrum):
    _ = plt.figure(figsize=(25, 15))
    _ = plt.plot(spectrum)


_ = """########################################################################
################################# CORE EXECUTION ##############################
############################################################################"""

# 1. Get the file path to an included audio example
# filename = librosa.example('nutcracker')
fname = 'piano_A_sharp'
output_path = convert_file(input_path=f'music_files/{fname}.mp3',
                           output_path=f'music_files/converted-{fname}.wav')

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
waveform, sampling_rate = librosa.load(output_path)

plot_waveform(waveform)

STFT_DB_waveform = librosa_plot_spectrogram(waveform)

# spectrum, row_freqs, col_mpoints = plot_spectrogram(waveform, sampling_rate)
# plot_entire_freq_hist(spectrum)

original_chroma, index_mapping = librosa_plot_chroma(waveform, sampling_rate)


{
    """
modified_spectrum = plot_modified_spectrogram(
    spectrum, masked=True, skip_modify=False, tuning=0.10)

modified_spectrum.transpose().mean(axis=1)


def weighted_periodogram_freq(modified_spectrum):
    usable_matrix = modified_spectrum.transpose()
    for t_gram in usable_matrix:
        for i in range(len(t_gram)):
            t_gram[i] *= i


_ = plot_modified_spectrogram(usable_matrix.transpose(), masked=True)
"""
    """ UNUSED ATM.
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)

plt.figure(figsize=(12, 8))
_ = plt.hist(beat_frames, bins=100)
"""
}
# EOF
