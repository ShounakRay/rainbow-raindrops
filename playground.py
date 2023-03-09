# @Author: shounak
# @Date:   2022-05-14T14:55:56-07:00
# @Email:  shounak@stanford.edu
# @Filename: playground.py
# @Last modified by:   shounak
# @Last modified time: 2022-05-24T00:55:32-07:00

import collections
import numpy as np
import subprocess
import librosa.display
import librosa
import math
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from music21 import note, stream
import mingus.core.notes as notes

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

Specific process:
1. Import file.
2. Get waveform and sampling rate
3. Produce spectrogram
4. Identify notes from spectrogram (use some probability)
5. Manually categorize RH and LH for first 5 (useful) seconds (our priors)
6. Based on priors and additional assumptions, separate hands for rest of piece
7. Generate visualizations + sheet music. Switch between minor and major, etc.
"""

_ = """########################################################################
################################ HYPERPARAMETERS ##############################
############################################################################"""

PIANO_RANGE = (27.5, 4186)
SENTINEL_OOB = -1

_ = """########################################################################
################################## DEFINTIONS #################################
############################################################################"""


def frequency_to_note(frequency):
    if frequency <= 0:
        return 'Impossible' + str(SENTINEL_OOB)
    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#',
             'A', 'A#', 'B']  # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2  # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = (
        'A', 4, 440)  # A4 = 440 Hz

    """
    calculate the distance to the known note since notes are spread evenly,
    going up a note will multiply bya constant so we can use log to know how
    many times a frequency was multiplied to get from the known note to our
    note this will give a positive integer value for notes higher than the
    known note, and a negative value for notes lower than it (and zero for the
    same note)
    """
    note_multiplier = OCTAVE_MULTIPLIER**(1 / len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = math.log(
        frequency_relative_to_known_note, note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = round(distance_from_known_note)

    """
    using the distance in notes and the octave and name of the known note,
    we can calculate the octave and name of our note
    NOTE: the "absolute index" doesn't have any actual meaning, since it
    doesn't care what its zero point is. it is just useful for calculation
    """
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * \
        len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(
        NOTES), note_absolute_index % len(NOTES)
    if not (note_octave >= 1 and note_octave <= 7):
        return 'Impossible' + str(SENTINEL_OOB)
    note_name = NOTES[note_index_in_octave]
    return note_name + str(note_octave)


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
    """LIBROSA METHOD."""
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
    return collections.OrderedDict(sorted(notes_captured.items(),
                                          reverse=True))


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


def util_normalize(data, range=(0, 1)):
    scaler = MinMaxScaler(feature_range=(range[0], range[1]))
    res = scaler.fit_transform(data.reshape(-1, 1))
    return [int(i) for i in res]


def util_valid_note(the_note):
    return notes.is_valid_note(the_note)


def util_rm_octave(the_full_note):
    return ''.join([i for i in the_full_note if not i.isdigit()])


def plot_note_series(series_of_notes):
    """ASSUMPTION: type list and all caps."""
    stream1 = stream.Stream()
    for p_note in all_unique_notes:
        if not util_valid_note(util_rm_octave(p_note)):
            print(f'{p_note} is an invalid note. Skipping.')
        n_obj = note.Note(p_note)
        stream1.append(n_obj)
    stream1.show()


_ = """########################################################################
################################# CORE EXECUTION ##############################
############################################################################"""

# IMPORTS
# filename = librosa.example('nutcracker')
fname = 'piano_A_sharp'
output_path = convert_file(input_path=f'music_files/{fname}.mp3',
                           output_path=f'music_files/converted-{fname}.wav')

# GET DATA
waveform, sampling_rate = librosa.load(output_path)

""" EXPLORATORY ANALYSIS """
plot_waveform(waveform)
STFT_DB_waveform = librosa_plot_spectrogram(waveform)

original_chroma, index_mapping = librosa_plot_chroma(waveform, sampling_rate)
""" END """

""" ACTUAL PROCESSING """
frequencies, times, spectrogram = signal.spectrogram(waveform, sampling_rate)

# Mapping of (0, 129) to actual (0, 10000) Hz markings
spectrogram_to_use = np.flipud(spectrogram)
spect_indices = np.array([i for i in range(len(spectrogram_to_use))])
normalized = util_normalize(
    spect_indices, (min(frequencies), max(frequencies)))
freq_mapping = dict(zip(spect_indices, normalized))

# Ignore all data outside of possible piano range
filtered_spectrogram = []
for i in range(len(spectrogram_to_use)):
    corrs_freq = freq_mapping.get(i)
    if corrs_freq >= PIANO_RANGE[0] and corrs_freq <= PIANO_RANGE[1]:
        filtered_spectrogram.append(spectrogram_to_use[i])
    else:
        filtered_spectrogram.append([0] * len(spectrogram_to_use[i]))
filtered_spectrogram = np.array(filtered_spectrogram)

# Per time interval, get max frequencies
transposed_spectrogram = filtered_spectrogram.T
all_maxes = []
remove_zeros = True
for p_gram in transposed_spectrogram:
    if remove_zeros:
        p_gram = [i for i in p_gram if i != 0.]
        if len(p_gram) == 0:
            continue    # TODO: Check if this is correct.
    # TODO: Incorporate some percentile check
    index_max = np.argmax(p_gram)   # This is constrained from (0, 129): freq.
    f_max = freq_mapping.get(index_max)
    all_maxes.append(f_max)

# {EXPLORATION} Plot all the max frequencies
plt.figure(figsize=(12, 8))
plt.plot(all_maxes[1:10])
_ = plt.hist(all_maxes, bins=100)

# Plot all the unique notes (no rests or anything)
max_notes = [i for i in map(
    frequency_to_note, all_maxes) if i != 'Impossible-1']
sorted(collections.Counter(max_notes).items(), key=lambda x: x[1])
all_unique_notes = set(max_notes)
plot_note_series(all_unique_notes)

# http://bspaans.github.io/python-mingus/doc/wiki/tutorialNote.html
# https://web.mit.edu/music21/doc/usersGuide/usersGuide_04_stream1.html
# TODO: Modularize function defintions
# TODO: De-sensitize max identification.
# IDEA: Time sig. could be an input for sheet-music. makes life easier.

# EOF
