# CST2216 Neural Networks Streamlit Project

## Project Overview
This project modularizes the UCLA Neural Networks Jupyter Notebook into a VS Code project and deploys it using Streamlit Cloud. The application predicts admission chance using neural network models.

## Dataset
- File: `Admission.csv`
- Target column: `Admit_Chance`
- Binary conversion rule: `1 if Admit_Chance >= 0.8 else 0`

## Project Structure
```text
cst2216-neural-networks-streamlit/
│
├── data/
│   └── Admission.csv
├── logs/
├── models/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visuals.py
│   └── utils.py
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md

## GitHub Repository
[GitHub Repo Link](https://github.com/SaraKhandakar/CST2216-Neural-Networks-Streamlit)

## Streamlit App Link
[Streamlit App Link](https://5fmywzpyq8h8mcm7bacngh.streamlit.app/)