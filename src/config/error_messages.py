ERROR_MSG: dict[str, str] = dict(
    # MODEL LOADING
    MODEL_NOT_LOADED="CLAP model is not loaded. "
    "Call load_model() before embedding operations.",
    PROCESSOR_NOT_LOADED="Processor or model is not loaded. "
    "Ensure load_model() was called.",
    TOKENIZER_NOT_LOADED="Tokenizer or model is not loaded. "
    "Ensure load_model() was called.",
    # INPUTS
    EMPTY_TEXT_INPUT="Text input is empty. Please provide a valid text prompt.",
    AUDIO_FILE_NOT_FOUND="Audio file not found at",
    INVALID_AUDIO_FORMAT="Unsupported audio format. "
    "Please upload a valid .wav or .mp3 file.",
    INVALID_TEXT_TYPE="Text input must be a string.",
    # EMBEDDING MODELS VALIDATION
    INVALID_EMBEDDING_MODEL="Invalid embedding model selected.",
    INVALID_AUDIO_EMBEDDING_MODE="Invalid audio embedding mode. Please use 'global' or 'frames'."
)
