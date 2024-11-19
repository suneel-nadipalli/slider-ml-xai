import streamlit as st
import sys
sys.path.append("..")

from utils.helper import *
from utils.ui.clf_components import *

def reset_classification_state():
    """Resets classification-related session state variables."""
    st.session_state.clf_selected_sample = None
    st.session_state.clf_slider_values = None
    st.session_state.clf_previous_proba = None
    st.session_state.clf_updated_proba = None

def display_clf():
    st.header("Classification")
    
    # Initialize session state variables for classification if not present
    if "clf_selected_sample" not in st.session_state:
        reset_classification_state()

    classification_option = st.radio(
        "Choose a classification setup:",
        ("Upload Files", "Use Preloaded Models/Datasets"),
        horizontal=True,
    )

    model, dataset = None, None

    if classification_option == "Upload Files":
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Upload your classification model (.pkl):", type=["pkl"], key="clf_model_file")
        with col2:
            dataset_file = st.file_uploader("Upload your classification dataset (.csv):", type=["csv"], key="clf_dataset_file")

        if model_file and dataset_file:
            if st.session_state.last_classification_model != model_file or st.session_state.last_classification_dataset != dataset_file:
                st.session_state.last_classification_model = model_file
                st.session_state.last_classification_dataset = dataset_file
                reset_classification_state()

            model = load_model(model_file)
            dataset = load_dataset(dataset_file)

    else:
        preloaded_option = st.selectbox(
            "Select a preloaded model and dataset:",
            [
                "Logistic Regressor + Titanic Dataset",
            ],
        )

        if preloaded_option == "Logistic Regressor + Titanic Dataset":
            model_path, dataset_path = "models/log_reg_titanic.pkl", "data/log_reg_titanic_test.csv"

            if st.session_state.last_classification_model != model_path or st.session_state.last_classification_dataset != dataset_path:
                st.session_state.last_classification_model = model_path
                st.session_state.last_classification_dataset = dataset_path
                reset_classification_state()

            model = load_model(model_path)
            dataset = load_dataset(dataset_path)

    if model and dataset is not None:
        if st.button("Load New Sample 🔄", key="clf_load_sample"):
            st.session_state.clf_selected_sample = dataset.sample(1).iloc[0]
            st.session_state.clf_slider_values = st.session_state.clf_selected_sample.to_dict()
            st.session_state.clf_previous_proba = model.predict_proba([list(st.session_state.clf_slider_values.values())])[0]
            st.session_state.clf_updated_proba = st.session_state.clf_previous_proba

        if st.session_state.clf_selected_sample is not None:
            sample = st.session_state.clf_selected_sample

            # Class Probabilities
            if hasattr(model, "predict_proba"):
                class_names = model.classes_
                st.subheader("Class Probabilities")
                display_split_rectangle(st.session_state.clf_previous_proba, class_names)

                st.subheader("Updated Class Probabilities")
                updated_sample = [st.session_state.clf_slider_values[feature] for feature in st.session_state.clf_slider_values.keys()]
                if hasattr(model, "predict_proba"):
                    st.session_state.clf_updated_proba = model.predict_proba([updated_sample])[0]
                    display_split_rectangle(st.session_state.clf_updated_proba, class_names)

            # Feature Importance
            elif hasattr(model, "feature_importances_"):
                st.markdown("### Feature Importances (Tree Model)")
                importances = model.feature_importances_
                for feature, importance in zip(dataset.columns, importances):
                    st.write(f"{feature}: {importance:.2f}")

            # Sliders for Feature Adjustment
            st.subheader("Adjust Features")
            cols = st.columns(len(sample))
            for col, feature in zip(cols, sample.index):
                with col:
                    st.markdown(f"<div style='text-align: center; font-size: 14px;'>{feature}</div>", unsafe_allow_html=True)
                    st.session_state.clf_slider_values[feature] = vertical_slider(
                        key=f"slider-{feature}",
                        default_value=sample[feature],
                        min_value=round(float(dataset[feature].min()), 2),
                        max_value=round(float(dataset[feature].max()), 2),
                    )
    else:
        st.warning("Click 'Load New Sample' to start.")

