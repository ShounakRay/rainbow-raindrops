# @Author: shounak
# @Date:   2022-05-18T04:28:49-07:00
# @Email:  shounak@stanford.edu
# @Filename: extracontent.py
# @Last modified by:   shounak
# @Last modified time: 2022-05-18T05:17:16-07:00

from pydub import AudioSegment
# import mutagen
import moviepy.editor as mp
import pyaudio
import wave

CODEC_MAPPING = {'mp4': 'libx264',
                 'ogv': 'libtheora',
                 'webm': 'libvpx',
                 'ogg': 'libvorbis',
                 'mp3': 'pcm_s16le',
                 'wav': 'libvorbis',
                 'm4a': 'libfdk_aac'}


def play_file(fname):
    """ SOURCE:
    https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
    """
    # define stream chunk
    chunk = 1024

    # open a wav format music
    f = wave.open(fname, "rb")
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()


def play_file(fname):
    """ SOURCE:
    https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python
    """
    # define stream chunk
    chunk = 1024

    # open a wav format music
    f = wave.open(fname, "rb")
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()


def video_to_wav(input_path, output_path=""):
    clip = mp.VideoFileClip(input_path)
    clip.audio.write_audiofile(output_path, codec=CODEC_MAPPING['wav'])


def convert_to_wav(input_path, output_path=""):
    _extension = input_path.split('.')[-1]
    output_path = input_path[:-3] + 'wav'
    if not output_path.startswith('converted-'):
        to_rep = output_path.split('/')[-1]
        output_path = output_path.replace(to_rep, 'converted-' + to_rep)

    if _extension == 'mp3':
        sound = AudioSegment.from_mp3(input_path)
        sound.export(output_path, format="wav")
    elif _extension == 'mov':
        video_to_wav(input_path, output_path)
    print(f"The new .wav file was saved at: {output_path}")

    return output_path


# def ipython_play_file(fname):
#     y, sr = librosa.load(fname)
#     Audio(data=y, rate=sr)

# EOF
