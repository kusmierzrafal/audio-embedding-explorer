# audio-embedding-explorer

Navigate to the project directory:
```bash
cd audio-embedding-explorer
```
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run main.py
```

## Development

### Code Quality with Ruff

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. 

**Install development dependencies:**
```bash
pip install -r requirements.txt
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