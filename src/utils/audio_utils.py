import io
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf


class AudioHelper:
    @staticmethod
    def load_audio(
        source: Path | io.BytesIO,
        target_sr: int = None,
        mono: bool = True,
        normalize: bool = True,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, int]:
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
    def segment_audio(
        waveform: np.ndarray, sr: int, segment_duration: float
    ) -> List[np.ndarray]:
        segment_length = int(sr * segment_duration)
        segments = []
        for start in range(0, waveform.shape[-1], segment_length):
            end = start + segment_length
            segment = waveform[start:end]
            segments.append(segment)
        return segments

    @staticmethod
    def process_audio(y: np.ndarray, sr: int, speed_rate=1.0, pitch_steps=0, noise_amount=0.0) -> np.ndarray:
        y_edited = y.copy()
        if speed_rate != 1.0:
            y_edited = librosa.effects.time_stretch(y_edited, rate=speed_rate)

        if pitch_steps != 0:
            y_edited = librosa.effects.pitch_shift(y_edited, sr=sr, n_steps=pitch_steps)

        if noise_amount > 0.0:
            noise = noise_amount * np.random.randn(len(y_edited))
            y_edited += noise

        return y_edited

    @staticmethod
    def samples_to_bytes(y: np.ndarray, sr: int) -> io.BytesIO:
        io_bytes = io.BytesIO()    
        sf.write(io_bytes, y, sr, format='WAV')
        return io_bytes.getvalue()