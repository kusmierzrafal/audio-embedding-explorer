# Changelog
All project changes will be documented in this file.

### Format:  
[version] - YYYY-MM-DD  
Sections: Added / Changed / Fixed / Notes  

---
### [0.9.0] - 2025-12-01
#### Fixed
- Reduced embeddings models to CLAP only for allowing to use on ARM devices
- Cleanup models managing logic
- Removed docker setup due to incompatibility with ARM devices for backend app image

---
### [0.8.0] - 2025-11-25
#### Added 
- Docker compose setup for easy local development and deployment

---

### [0.7.0] - 2025-10-28
#### Added
- NSynth Dataset (2017) sample for prototyping and student project work
- Referenced by course instructor as good resource for experimentation
- Source: https://magenta.withgoogle.com/datasets/nsynth#files
- Location: assets/nsynth-test/

---
 
### [0.6.0] - 2025-10-25
#### Added 
- OpenL3 embedder for audio embedding extraction
- MERT embedder for audio embedding extraction
- `EmbeddersManager` service to manage different embedding models

#### Changed
- Need to downgrade Python to 3.11 due to OpenL3 compatibility issues
- Pair Analysis view to support OpenL3 and MERT embedders for audio embedding extraction
- Embedders are lazy loaded when first requested
- New .env variables for MERT (check in `src/config/env_keys.py`)
- Splitted `BaseEmbedder` into `AudioEmbedder` and `TextEmbedder` abstract classes
- Move audio logic to `audio_utils.py` helper

---

### [0.5.0] - 2025-10-24
#### Added
- View `similarity_ranking_view.py` that creates ranking of audio and text similarity

### Notes
- Tab Text → Audio - user inputs text description, uploads multiple audio files 
and receives a table with ranked similarity and audio previews.
- Tab Audio → Text - user uploads one audio file, inputs multiple text descriptions 
and gets a table with ranked similarity.

---

### [0.4.0] - 2025-10-24
#### Added
- TSNE & UMAP reduce dimensions methods in reduce_embeddings_dimensionality function

#### Changed
- PCA plotting function

#### Removed
- CLAP heatmap plot due to lack of analytical value and poor interpretability

---

### [0.3.0] - 2025-10-23
#### Added
- GitHub Actions workflow for Ruff linting and formatting checks on pull requests
- Ruff configuration in `pyproject.toml` with Python 3.13 target and basic linting rules
- Dependencies in `requirements.txt` including Ruff 0.8.4
- Comprehensive development documentation in README with local Ruff usage instructions

#### Changed
- Updated README with development section including code quality guidelines
- Added CI/CD documentation explaining the automated PR workflow

#### Notes
Pull requests now require passing Ruff checks before merging. This ensures consistent code quality and formatting across the project.

---

### [0.2.0] - 2025-10-22
#### Added
- BaseEmbedder, ClapEmbedder
- Computations for audio & text inputs
- Basic metrics and visualisations

#### Changed
- Refactored Application class to preload models and store them in st.session_state

---

### [0.1.0] - 2025-10-20
#### Added
- Implemented full project structure (`src/`, `app/`, `config/`, `domain/`, `models/`, `ui/`, `utils/`, `assets/`)
- Created functional Streamlit UI with working navigation
- Added views for all major screens (Home, Pair Analysis, Model Comparison, Similarity Ranking, Pseudo-captioning, Local Database)

#### Notes
This version introduces a complete functional skeleton of the application.
No real embedding computations yet. UI and structure ready for integration.

---
