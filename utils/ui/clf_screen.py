import streamlit as st

import sys, random

sys.path.append("..")

from utils.helper import *

from utils.ui.clf_components import *

def display_clf():
    st.header("Classification")
    # Option to choose preloaded models/datasets or upload files
    classification_option = st.radio(
        "Choose a classification setup:",
        ("Upload Files", "Use Preloaded Models/Datasets"),
        horizontal=True,
    )

    if classification_option == "Upload Files":
        # File upload
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Upload your classification model (.pkl):", type=["pkl"])
        with col2:
            dataset_file = st.file_uploader("Upload your classification dataset (.csv):", type=["csv"])
    else:
        # Preloaded models and datasets
        preloaded_option = st.selectbox(
            "Select a preloaded model and dataset:",
            [
                "Logistic Regressor + Titanic Dataset",
            ],
        )

        if preloaded_option == "Logistic Regressor + Titanic Dataset":
            st.session_state.selected_sample = None
            st.session_state.slider_values = None
            model = load_model("models/log_reg_titanic.pkl")
            dataset = load_dataset("data/log_reg_titanic_test.csv")

    if "model" in locals() or ("model_file" in locals() and dataset_file):
        if classification_option == "Upload Files" and model_file and dataset_file:
            model = load_model(model_file)
            dataset = load_dataset(dataset_file)

        if st.button("Load New Sample 🔄",key="clf_key"):
            # Load new sample and update state
            st.session_state.selected_sample = dataset.sample(1).iloc[0]
            st.session_state.slider_values = st.session_state.selected_sample.to_dict()
            st.session_state.previous_proba = model.predict_proba([list(st.session_state.slider_values.values())])[0]
            st.session_state.updated_proba = st.session_state.previous_proba

        if st.session_state.selected_sample is not None:
            sample = st.session_state.selected_sample

            # --- Class Probabilities ---
            if hasattr(model, "predict_proba"):
                class_names = model.classes_

                st.subheader("Class Probabilities")
                display_split_rectangle(st.session_state.previous_proba, class_names)

                st.subheader("Updated Class Probabilities")
                updated_sample = [st.session_state.slider_values[feature] for feature in st.session_state.slider_values.keys()]
                if hasattr(model, "predict_proba"):
                    st.session_state.updated_proba = model.predict_proba([updated_sample])[0]
                    display_split_rectangle(st.session_state.updated_proba, class_names)

            elif hasattr(model, "feature_importances_"):
                st.markdown("### Feature Importances (Tree Model)")
                importances = model.feature_importances_
                for feature, importance in zip(dataset.columns, importances):
                    st.write(f"{feature}: {importance:.2f}")

            # --- Sliders for Feature Adjustment ---
            st.subheader("Adjust Features")
            cols = st.columns(len(sample))
            for col, feature in zip(cols, sample.index):
                with col:
                    st.markdown(f"<div style='text-align: center; font-size: 14px;'>{feature}</div>", unsafe_allow_html=True)
                    st.session_state.slider_values[feature] = vertical_slider(
                        key=f"slider-{feature}",
                        default_value=sample[feature],
                        min_value=round(float(dataset[feature].min()), 2),
                        max_value=round(float(dataset[feature].max()), 2),
                    )

            # --- Updated Probabilities ---


    else:
        st.warning("Click 'Load New Sample' to start.")