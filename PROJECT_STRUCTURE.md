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
