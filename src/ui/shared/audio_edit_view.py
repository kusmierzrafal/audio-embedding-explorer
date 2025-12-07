import io

import streamlit as st

from src.utils.audio_utils import AudioHelper


class AudioEditView:
    def __init__(self, audio_name: str, audio_bytes: io.BytesIO, sr: int):
        self._audio_name = audio_name
        self._sr = sr
        self._state_prefix = f"audio_edit_{audio_name}"

        if self._state_prefix not in st.session_state:
            st.session_state[self._state_prefix] = AudioEditCache(audio_bytes, sr)

        self.latest_y = st.session_state[self._state_prefix].latest_y

    def render(self) -> None:
        st.markdown(f"##### [{self._audio_name}]")

        cache: AudioEditCache = st.session_state[self._state_prefix]
        st.audio(cache.audio_bytes, format="audio/wav")

        col1, col2, col3 = st.columns(3)

        speed_rate = col1.slider(
            "Speed rate", min_value=0.5, max_value=2.0, value=cache.speed_rate, step=0.1
        )

        pitch_steps = col2.slider(
            "Pitch (semitones)",
            min_value=-12,
            max_value=12,
            value=cache.pitch_steps,
            step=1,
        )

        noise_amount = col3.slider(
            "Noise level",
            min_value=0.0,
            max_value=0.1,
            value=cache.noise_amount,
            step=0.01,
            help="Gauss noise added to the signal (standard deviation).",
        )
        button_disabled = (
            speed_rate == cache.speed_rate
            and pitch_steps == cache.pitch_steps
            and noise_amount == cache.noise_amount
        )

        if st.button("Apply edits", disabled=button_disabled):
            with st.spinner("Processing audio..."):
                processed_y = AudioHelper.process_audio(
                    cache.original_y,
                    sr=self._sr,
                    speed_rate=speed_rate,
                    pitch_steps=pitch_steps,
                    noise_amount=noise_amount,
                )
                cache.audio_bytes = AudioHelper.samples_to_bytes(
                    processed_y, sr=self._sr
                )
                cache.speed_rate = speed_rate
                cache.pitch_steps = pitch_steps
                cache.noise_amount = noise_amount
                cache.latest_y = processed_y

                st.success("Audio processed successfully!")
                st.rerun()


class AudioEditCache:
    def __init__(self, audio_bytes: io.BytesIO, sr: int):
        self.audio_bytes = audio_bytes
        self.original_y, _ = AudioHelper.load_audio(
            source=audio_bytes,
            target_sr=sr,
            mono=True,
            normalize=True,
        )
        self.latest_y = self.original_y
        self.speed_rate = 1.0
        self.pitch_steps = 0
        self.noise_amount = 0.0
