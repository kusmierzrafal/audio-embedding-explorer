# Changelog
All project changes will be documented in this file.

### Format:  
[version] - YYYY-MM-DD  
Sections: Added / Changed / Fixed / Notes  

---

### [0.15.0] - 2025-12-27
#### Added
- Interactive PCA and UMAP visualizations in Model Comparison View using Plotly
- Click-to-select functionality on scatter plot points
- Audio preview panel for selected points with file details
- Hover tooltips displaying audio names and coordinates on plots

#### Changed
- Replaced static matplotlib plots with interactive Plotly charts
- Enhanced user experience with point selection and audio playback integration

---

### [0.14.0] - 2025-12-27
#### Added
- Local Database View with semantic search functionality
- Text → Audio search mode using CLAP models for semantic queries
- Audio → Audio similarity search mode compatible with all embedding models
- Cosine similarity ranking with configurable top-K results
- Interactive audio preview and database integration for search results
- Support for both text descriptions and audio-based similarity queries

---

### [0.13.0] - 2025-12-27
#### Added
- Database Management view for interactive inspection, preview, and deletion of audio files and embeddings

---

### [0.12.0] - 2025-12-26
#### Added
- Database integration with SQLite backend for persistent data storage
- Embedding caching system to store and reuse computed embeddings across all views
- Database operations for audio files and embeddings with SHA256-based deduplication
- Save to database functionality in all audio input views (Embeddings Playground, Similarity Ranking, Model Comparison, Pseudo Captioning)
- Automatic embedding cache lookup and storage for significant performance improvements on repeat operations

#### Changed
- All views now use centralized embedding cache through `DbManager.get_or_compute_audio_embedding()`
- Model Comparison view now includes scrollable audio file list with fixed height for better UX
- Database-backed audio file selection available as alternative to file upload in multiple views

#### Fixed
- Improved handling of different file object types (BytesIO vs uploaded files) across views
- Better error handling for database operations with automatic fallback to direct computation

---

### [0.11.0] - 2025-12-23
#### Added
- Model Comparison view for side-by-side comparison of audio embedding models
- Audio file upload and managed audio list with playback and deletion
- Per-model PCA and UMAP visualizations for generated embeddings
- NSynth-test dataset available in assets for audio embedding experiments

---

### [0.10.0] - 2025-12-20
#### Changed
- Replaced mocked Pseudo-captioning view with a real implementation based on CLAP models (music-level & sound-level)
- Pseudo-captioning now uses uploaded audio and caption data loaded from JSON assets
- Pseudo-captioning similarity computation now uses audio embedding vs precomputed caption embeddings

---

### [0.9.1] - 2025-12-19
#### Fixed
- Prevent access to non-Home views when no models are loaded
- Fix crashes caused by empty model selections
- Improve device detection (auto CPU / CUDA fallback)
- Safer optional imports for embedding backends

#### Changed
- Centralized model availability checks in application flow
- Improved UX messaging when no models are available

---

### [0.9.0] - 2025-12-07
#### Fixed
- Cleanup models managing logic (handling import errors)
- Add dashboard for models managing
- Added audio edit functionality
- Refactor pair analysis view (changed to embeddings playground)
- Removed docker setup due to incompatibility with ARM devices for backend app image
- Migrating from raw pip to uvicorn for backend app serving

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
