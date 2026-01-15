# audio-embedding-explorer

### Local Development Setup
Navigate to the project directory:
```bash
cd audio-embedding-explorer
```
Create and activate a virtual environment for python 3.11:
```bash
py -3.11 -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
```
Install dependencies:
```bash
pip install uv
uv sync
```
Run the Streamlit app:
```bash
uv run streamlit run main.py
```

Access the Streamlit app at `http://localhost:8051`.

## GPU/CPU Configuration

The application uses CPU by default for audio embedding models. To enable GPU acceleration:

1. **Create `.env` file** (if not already present):
   ```bash
   DEVICE=cpu
   ```

2. **Switch to GPU** by changing the device setting:
   ```bash
   DEVICE=cuda
   ```

**Requirements for GPU:**
- CUDA-compatible GPU
- PyTorch with CUDA support (automatically installed with `uv sync`)

## Development

### Code Quality with Ruff

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. 

**Install development dependencies:**
```bash
uv sync
```

**Run Ruff checks locally:**
```bash
# Check for linting issues
ruff check .

# Check formatting (without making changes)
ruff format --check .

# Auto-fix linting issues where possible
ruff check --fix .

# Format code automatically
ruff format .
```

**Run all checks (recommended before committing):**
```bash
ruff check . && ruff format --check .
```

### CI/CD

Pull requests are automatically checked with Ruff via GitHub Actions. The workflow:
- Runs linting checks with inline annotations
- Verifies code formatting compliance
- Must pass before merging is allowed