# Changelog
All project changes will be documented in this file.

### Format:  
[version] - YYYY-MM-DD  
Sections: Added / Changed / Fixed / Notes  

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
