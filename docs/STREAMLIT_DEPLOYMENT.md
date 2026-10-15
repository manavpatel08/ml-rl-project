# Streamlit Hosting and Deployment Notes

## Deployment target

Use Streamlit Community Cloud from a GitHub repository for the initial public demo.

Official documentation:
- Deploy an app: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- App file organization: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- Dependency management: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Requirements

- Put `app.py` at the repository root for the simplest deployment path.
- Put `requirements.txt` at the repository root.
- Keep `.streamlit/config.toml`, if needed, at `.streamlit/config.toml`.
- Use the same Python version locally and on the hosted app when possible.
- Include only dependencies that the app actually needs.
- Keep the core environment headless; do not depend on opening a desktop window.
- Test PyTorch installation and CPU inference/training in the target Linux environment.
- Keep hosted training bounded. Do not expose an unlimited training button.
- Do not assume files written to a hosted app's local filesystem are durable across restarts. If persistence is required later, choose an explicit storage approach.

## Suggested local setup

Create and activate a virtual environment, install dependencies, then run tests and the app.

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

The exact requirements should be generated only after implementation and verified in the chosen environment. Avoid blindly pinning incompatible versions.
