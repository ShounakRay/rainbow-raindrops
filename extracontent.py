# @Author: shounak
# @Date:   2022-05-18T04:28:49-07:00
# @Email:  shounak@stanford.edu
# @Filename: extracontent.py
# @Last modified by:   shounak
# @Last modified time: 2022-05-20T04:20:06-07:00

""" ROBUST FILE IO """
# from pydub import AudioSegment
# # import mutagen
# import moviepy.editor as mp
# import pyaudio
# import wave
#
# CODEC_MAPPING = {'mp4': 'libx264',
#                  'ogv': 'libtheora',
#                  'webm': 'libvpx',
#                  'ogg': 'libvorbis',
#                  'mp3': 'pcm_s16le',
#                  'wav': 'libvorbis',
#                  'm4a': 'libfdk_aac'}
#
#
# def play_file(fname):
#     """ SOURCE:
#     https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
#     """
#     # define stream chunk
#     chunk = 1024
#
#     # open a wav format music
#     f = wave.open(fname, "rb")
#     # instantiate PyAudio
#     p = pyaudio.PyAudio()
#     # open stream
#     stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
#                     channels=f.getnchannels(),
#                     rate=f.getframerate(),
#                     output=True)
#     # read data
#     data = f.readframes(chunk)
#
#     # play stream
#     while data:
#         stream.write(data)
#         data = f.readframes(chunk)
#
#     # stop stream
#     stream.stop_stream()
#     stream.close()
#
#     # close PyAudio
#     p.terminate()
#
#
# def play_file(fname):
#     """ SOURCE:
#     https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
#     """
#     # define stream chunk
#     chunk = 1024
#
#     # open a wav format music
#     f = wave.open(fname, "rb")
#     # instantiate PyAudio
#     p = pyaudio.PyAudio()
#     # open stream
#     stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
#                     channels=f.getnchannels(),
#                     rate=f.getframerate(),
#                     output=True)
#     # read data
#     data = f.readframes(chunk)
#
#     # play stream
#     while data:
#         stream.write(data)
#         data = f.readframes(chunk)
#
#     # stop stream
#     stream.stop_stream()
#     stream.close()
#
#     # close PyAudio
#     p.terminate()
#
#
# def video_to_wav(input_path, output_path=""):
#     clip = mp.VideoFileClip(input_path)
#     clip.audio.write_audiofile(output_path, codec=CODEC_MAPPING['wav'])
#
#
# def convert_to_wav(input_path, output_path=""):
#     _extension = input_path.split('.')[-1]
#     output_path = input_path[:-3] + 'wav'
#     if not output_path.startswith('converted-'):
#         to_rep = output_path.split('/')[-1]
#         output_path = output_path.replace(to_rep, 'converted-' + to_rep)
#
#     if _extension == 'mp3':
#         sound = AudioSegment.from_mp3(input_path)
#         sound.export(output_path, format="wav")
#     elif _extension == 'mov':
#         video_to_wav(input_path, output_path)
#     print(f"The new .wav file was saved at: {output_path}")
#
#     return output_path

""" PLAYING LIBROSA SOUND """
# def ipython_play_file(fname):
#     y, sr = librosa.load(fname)
#     Audio(data=y, rate=sr)

""" Own spectrogram stuff """
# def plot_modified_spectrogram(spectrum, PIANO_RANGE=PIANO_RANGE, masked=True,
#                               skip_modify=False, tuning=0.1):
#     spectrum = np.flipud(spectrum)
#     _ = plt.figure(figsize=(18, 15))
#     if not skip_modify:
#         spectrum = spectrum ** tuning
#         if masked:
#             spectrum = spectrum[PIANO_RANGE[0]:PIANO_RANGE[1], :]
#     _ = sns.heatmap(spectrum)
#     return spectrum
# modified_spectrum = plot_modified_spectrogram(
#     spectrum, masked=True, skip_modify=False, tuning=0.10)
#
# modified_spectrum.transpose().mean(axis=1)
#
#
# def weighted_periodogram_freq(modified_spectrum):
#     usable_matrix = modified_spectrum.transpose()
#     for t_gram in usable_matrix:
#         for i in range(len(t_gram)):
#             t_gram[i] *= i
#
#
# _ = plot_modified_spectrogram(usable_matrix.transpose(), masked=True)
# """
#     """ UNUSED ATM.
# # 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
#
# print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
#
# # 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)
#
# plt.figure(figsize=(12, 8))
# _ = plt.hist(beat_frames, bins=100)
#
# > Yet another spectrogram plot
# _ = plt.figure(figsize=(12, 30))
# _ = plt.pcolormesh(times, frequencies, spectrogram)
# # plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# _ = plt.figure(figsize=(12, 30))
# _ = sns.heatmap(np.flipud(spectrogram))

# def plot_entire_freq_hist(spectrum):
#     _ = plt.figure(figsize=(25, 15))
#     _ = plt.plot(spectrum)
# def plot_spectrogram(waveform, sampling_rate):
#     fig, ax = plt.subplots(figsize=(18, 15))
#     cmap = plt.get_cmap('viridis')
#     spectrum, row_freqs, col_mpoints, cax = plt.specgram(
#         waveform, Fs=sampling_rate, cmap=cmap, mode='magnitude', NFFT=256,
#         scale='dB')
#     plt.xlabel('Time')
#     plt.ylabel('Frequency (Hz)')
#     cbar = fig.colorbar(cax)
#     cbar.set_label('Amplitude', rotation=270)
#     return spectrum, row_freqs, col_mpoints
# spectrum, row_freqs, col_mpoints = plot_spectrogram(waveform, sampling_rate)
# plot_entire_freq_hist(spectrum)

""" FILE CHECKING """
# def isfile(fname):
#     return os.path.isfile(fname)

# EOF
