from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram
from madmom.audio.signal import FramedSignal
from torch.utils.data import Dataset
from madmom.io import midi
import numpy as np
import torch
import joblib
memory = joblib.memory.Memory('./joblib_cache', mmap_mode='r', verbose=1)


def get_y_from_file(midifile, n_frames, dt):
    pattern = midi.MIDIFile(midifile)

    y = np.zeros((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes:
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))

        # even if the event was too short, always produce a label!
        if frame_start == frame_end:
            frame_end += 1
        label = pitch - 21
        y[frame_start:frame_end, label] = 1

    return y


@memory.cache
def get_xy_from_file(audiofilename, midifilename):
    audio_options = dict(
        num_channels=1,
        sample_rate=44100,
        filterbank=LogarithmicFilterbank,
        frame_size=4096,
        fft_size=4096,
        hop_size=441 * 4,  # 25 fps
        num_bands=48,
        fmin=30,
        fmax=8000.0,
        fref=440.0,
        norm_filters=True,
        unique_filters=True,
        circular_shift=False,
        norm=True
    )

    dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])
    x = LogarithmicFilteredSpectrogram(audiofilename, **audio_options)
    y = get_y_from_file(midifilename, len(x), dt)

    return x, y


class OneSequenceDataset(Dataset):
    def __init__(self, audiofilename, midifilename, start_end=None):
        self.metadata = dict(
            audiofilename=audiofilename,
            midifilename=midifilename
        )
        x, y = get_xy_from_file(audiofilename, midifilename)
        if start_end is not None:
            start, end = start_end
            x = x[start:end]
            y = y[start:end]

        self.x = FramedSignal(
            x,
            frame_size=5,
            hop_size=1,
            origin='center'
        )
        self.y = y
        _, self.w, self.h = self.x.shape

    def __getitem__(self, index):
        tx = torch.FloatTensor(self.x[index].reshape(1, self.w, self.h))
        ty = torch.FloatTensor(self.y[index])
        return tx, ty

    def __len__(self):
        return len(self.x)


def get_sequences(filenames, start_end=None):
    sequences = []
    for audiofilename, midifilename in (p.split(',') for p in filenames):
        sequences.append(OneSequenceDataset(audiofilename, midifilename, start_end))
    return sequences
