# Changelog
All project changes will be documented in this file.

### Format:  
[version] - YYYY-MM-DD  
Sections: Added / Changed / Fixed / Notes  

---

### [0.2.0] - 2025-10-23
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

### [0.1.0] - 2025-10-20
#### Added
- Implemented full project structure (`src/`, `app/`, `config/`, `domain/`, `models/`, `ui/`, `utils/`, `assets/`)
- Created functional Streamlit UI with working navigation
- Added views for all major screens (Home, Pair Analysis, Model Comparison, Similarity Ranking, Pseudo-captioning, Local Database)

#### Notes
This version introduces a complete functional skeleton of the application.
No real embedding computations yet. UI and structure ready for integration.

---

### [0.2.0] - 2025-10-22
#### Added
- BaseEmbedder, ClapEmbedder
- Computations for audio & text inputs
- Basic metrics and visualisations

#### Changed
- Refactored Application class to preload models and store them in st.session_state

---
