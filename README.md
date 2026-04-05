# CST2216 Neural Networks Streamlit Project

## Project Overview

This project modularizes the UCLA Neural Networks Jupyter Notebook into a VS Code project and deploys it using Streamlit Cloud. The application predicts admission chance using neural network models.

## Dataset
File: Admission.csv
Target column: Admit_Chance
Binary conversion rule: 1 if Admit_Chance >= 0.8 else 0

## Project Structure

cst2216-neural-networks-streamlit/

data/
Admission.csv
logs/
models/
src/
init.py
config.py
data_loader.py
preprocessing.py
train.py
evaluate.py
predict.py
visuals.py
utils.py
app.py
requirements.txt
runtime.txt
.gitignore
README.md

## Features
Dataset preview and explanation
Binary classification using threshold 0.8
Neural Network using MLPClassifier
Model comparison (default vs tanh)
Accuracy, confusion matrix, classification report
Loss curve visualization
Interactive prediction

## How to Run Locally
Create virtual environment
python -m venv venv
Activate environment
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run app
streamlit run app.py

## Logging and Error Handling
Logging using Python logging module
Logs stored in logs/app.log
Error handling using try-except

## Links

GitHub Repository:
https://github.com/SaraKhandakar/CST2216-Neural-Networks-Streamlit

Streamlit App:
https://5fmywzpyq8h8mcm7bacngh.streamlit.app/

Author

Shara Khandakar