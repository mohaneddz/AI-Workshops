# Algerian Laptop Market Challenge (Part 1)

This challenge has a simple workflow:
1. preprocess the raw data
2. run `train.py`

## 1) Setup

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 2) Preprocess the data (required)

Open the notebook:
- `tasks/01_Preprocessing.ipynb`

Input files are in `data/`:
- `data/data.csv`
- `data/cpu_passmark.csv`
- `data/gpu_passmark.csv`

At the end of preprocessing, export the final dataset exactly as:
- `data/data_preprocessed.csv`

`train.py` reads only this file.

## 3) Train models

Run:
```bash
python train.py
```

This trains all available models and writes artifacts to `models/`.

## 4) Run the web app (optional)

```bash
python -m app.main
```

Open:
- `http://127.0.0.1:1234`

## Notes

- `tasks/02_EDA.ipynb` is optional for exploration and analysis.
- `RUN.txt` has the same flow in short form.
- Instructor reference notebooks are in `../Solutions/AlgerianLaptopAnalysis/`.
