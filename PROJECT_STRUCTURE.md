## Project structure info:

* .streamlit/ - configuration for streamlit
* assets/ - static resources ex. demo audio files, icons
* dataset/ - local data used for model testing
* src/ - source directory with all app code 
  * src/application.py - central controller
  * src/config/ - static configs and constants - no executable logic
  * src/domain/ - core logic of the app - e.g. embedding computations, model loading, metrics, visualisation
  * src/models/ - data models e.g. enums
  * src/ui/ - all streamlit views (pages/screens)
  * src/utils/ - helper functions that support the rest of the project, independent of domain logic
* main.py - entry point of the app, executed when *streamlit run main.py* is run
* requirements.txt - all required dependencies


Proposed file structure based on branch Environment_Init src/app.py contents:

| Function                   | Target File                             | Layer   |
|-----------------------------|------------------------------------------|----------|
| `resolve_paths()`           | `src/utils/path_utils.py`               | Utility  |
| `load_env_variables()`      | `src/utils/env_utils.py`                | Utility  |
| `ensure_clap_exists()`      | `src/domain/models/clap_loader.py`      | Domain   |
| `load_clap_model()`         | `src/domain/models/clap_loader.py`      | Domain   |
| `calculate_text_embedding()`| `src/domain/embeddings.py`              | Domain   |
| `calculate_audio_embedding()`| `src/domain/embeddings.py`             | Domain   |
