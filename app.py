import streamlit as st
import pandas as pd

from src.config import DATA_PATH, DEFAULT_MODEL_NAME, TANH_MODEL_NAME
from src.utils import setup_logging, format_prediction_label
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_model
from src.visuals import plot_confusion_matrix, plot_loss_curve
from src.predict import prepare_single_input, make_prediction


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="UCLA Neural Networks App",
    page_icon="🧠",
    layout="wide"
)

setup_logging()

st.title("UCLA Admission Prediction using Neural Networks")
st.write("This app follows the notebook logic exactly and compares two MLPClassifier models.")


# -----------------------------
# Load and preprocess data
# -----------------------------
@st.cache_data
def get_processed_data():
    df = load_data(DATA_PATH)
    return df


@st.cache_resource
def get_training_artifacts():
    df = load_data(DATA_PATH)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    models = train_models(X_train_scaled, y_train)

    # rebuild processed dataframe to get final training columns for prediction
    df_copy = df.copy()
    df_copy["Admit_Chance"] = (df_copy["Admit_Chance"] >= 0.8).astype(int)
    df_copy = df_copy.drop(columns=["Serial_No"])
    df_copy["University_Rating"] = df_copy["University_Rating"].astype("object")
    df_copy["Research"] = df_copy["Research"].astype("object")
    clean_data = pd.get_dummies(
        df_copy,
        columns=["University_Rating", "Research"],
        dtype=int
    )
    X = clean_data.drop(columns=["Admit_Chance"])
    training_columns = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, models, training_columns


df = get_processed_data()
X_train_scaled, X_test_scaled, y_train, y_test, scaler, models, training_columns = get_training_artifacts()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Choose a model")
selected_model_name = st.sidebar.selectbox(
    "Model",
    [DEFAULT_MODEL_NAME, TANH_MODEL_NAME]
)

selected_model = models[selected_model_name]


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset", "Model Evaluation", "Loss Curve", "Prediction"]
)


# -----------------------------
# Tab 1: Dataset
# -----------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Target Conversion Logic")
    st.write("Admit_Chance is converted to binary:")
    st.code("Admit_Chance = 1 if Admit_Chance >= 0.8 else 0")


# -----------------------------
# Tab 2: Evaluation
# -----------------------------
with tab2:
    st.subheader(f"Evaluation - {selected_model_name}")

    results = evaluate_model(selected_model, X_test_scaled, y_test)

    st.metric("Accuracy", f"{results['accuracy']:.4f}")

    st.subheader("Classification Report")
    st.text(results["classification_report"])

    st.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(selected_model, X_test_scaled, y_test)
    st.pyplot(fig_cm)


# -----------------------------
# Tab 3: Loss Curve
# -----------------------------
with tab3:
    st.subheader(f"Loss Curve - {selected_model_name}")
    fig_loss = plot_loss_curve(selected_model, selected_model_name)
    st.pyplot(fig_loss)


# -----------------------------
# Tab 4: Prediction
# -----------------------------
with tab4:
    st.subheader("Predict Admission Chance")

    gre_score = st.slider("GRE Score", 260, 340, 320)
    toefl_score = st.slider("TOEFL Score", 0, 120, 100)
    university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    sop = st.slider("SOP (Statement of Purpose)", 1.0, 5.0, 3.0, 0.5)
    lor = st.slider("LOR (Letter of Recommendation)", 1.0, 5.0, 3.0, 0.5)
    cgpa = st.slider("CGPA", 0.0, 10.0, 8.0, 0.1)
    research = st.selectbox("Research", [0, 1])

    if st.button("Predict"):
        user_input = prepare_single_input(
            gre_score=gre_score,
            toefl_score=toefl_score,
            university_rating=university_rating,
            sop=sop,
            lor=lor,
            cgpa=cgpa,
            research=research,
            training_columns=training_columns,
        )

        prediction, probability = make_prediction(selected_model, scaler, user_input)

        st.success(f"Prediction: {format_prediction_label(prediction)}")
        st.write(f"Probability of class 0: {probability[0]:.4f}")
        st.write(f"Probability of class 1: {probability[1]:.4f}")