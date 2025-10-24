import io
from pathlib import Path
import numpy as np
import torch
import librosa
from typing import Tuple, List

class AudioHelper:
    @staticmethod
    def load_audio(source: Path | io.BytesIO,
        target_sr: int = None,
        mono: bool = True,
        normalize: bool = True,
        dtype: np.dtype = np.float32) -> Tuple[np.ndarray, int]:

        waveform, sr = librosa.load(source, sr=target_sr, mono=mono)
        if normalize:
            waveform = AudioHelper.normalize_audio(waveform)
        return waveform.astype(dtype), sr

    @staticmethod
    def normalize_audio(waveform: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    @staticmethod
    def segment_audio(waveform: np.ndarray,
                      sr: int,
                      segment_duration: float) -> List[np.ndarray]:
        segment_length = int(sr * segment_duration)
        segments = []
        for start in range(0, waveform.shape[-1], segment_length):
            end = start + segment_length
            segment = waveform[start:end]
            segments.append(segment)
        return segments

    @staticmethod
    def to_tensor(waveform: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(waveform)
