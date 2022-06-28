import torch
from pydub import AudioSegment
import numpy as np
import librosa

def to_array(audio : AudioSegment, to_torch : bool = False) -> np.array:
    '''
    Convert AudioSegment into Numpy arrays

    Input:
    - audio : AudioSegment Object
    '''

    # Get slice of audiosegment object
    samples = audio.get_array_of_samples()
    array = np.array(samples, dtype = np.float32)

    # Reshape with channel information
    outshape = (-1, audio.channels)
    array = array.reshape(outshape).T
    if audio.channels == 1:
        array = array.squeeze(0)

    # Normalise to (-1,+1) using sample_width (default: 2)
    normalization = 1 << (8 * audio.sample_width - 1)
    array /= normalization

    if to_torch:
        array = torch.from_numpy(array)

    return array

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def preprocess_audio(wav_path):
    wav = AudioSegment.from_file(wav_path)
    normalized_wav = match_target_amplitude(sound=wav, target_dBFS=-10.0)
    normalized_wav = to_array(normalized_wav)
    processed_wav = torch.from_numpy(librosa.effects.preemphasis(normalized_wav)).unsqueeze(0)
    return processed_wav