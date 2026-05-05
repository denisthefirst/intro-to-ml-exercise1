# Project Title

Brief description of what this project does.

---

## Prerequisites
- Python 3.8 or higher  
- Git  

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/denisthefirst/intro-to-ml-exercise1.git
cd <your-project-folder>
```

---

### 2. Create and Activate Virtual Environment

#### macOS / Linux
```bashff
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (Command Prompt)
```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 3. Run the Project

Make sure your environment is active (`venv`).

```bash
python main.py
```

---

### 4. Deactivate

When finished:

```bash
deactivate
```

---

## Developer Notes

Update requirements before committing new packages:

```bash
pip freeze > requirements.txt
```