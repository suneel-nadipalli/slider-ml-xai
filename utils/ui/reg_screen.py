import os, sys

sys.path.append("..")

import random

from utils.helper import *

from utils.ui.reg_components import *

def display_reg():
    st.header("Regression")
    # Option to choose preloaded models/datasets or upload files
    regression_option = st.radio(
        "Choose a regression setup:",
        ("Upload Files", "Use Preloaded Models/Datasets"),
        horizontal=True,
    )

    if regression_option == "Upload Files":
        # File upload
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Upload your model (.pkl):", type=["pkl"])
        with col2:
            dataset_file = st.file_uploader("Upload your dataset (.csv):", type=["csv"])
    else:
        # Preloaded models and datasets
        preloaded_option = st.selectbox(
            "Select a preloaded model and dataset:",
            [
                "Linear Regressor + House Prices Dataset",
                "Decision Tree Regressor + Auto MPG Dataset",
            ],
        )

        if preloaded_option == "Linear Regressor + House Prices Dataset":
            st.session_state.selected_sample = None
            st.session_state.slider_values = None
            model = load_model("models/reg_house_price.pkl")  # Path to pre-saved model
            dataset = load_dataset("data/reg_house_price_test.csv")  # Path to pre-saved dataset
        elif preloaded_option == "Decision Tree Regressor + Auto MPG Dataset":
            st.session_state.selected_sample = None
            st.session_state.slider_values = None
            model = load_model("models/reg_tree_miles.pkl")
            dataset = load_dataset("data/reg_tree_miles_test.csv")

    # If files are uploaded or preloaded models are selected
    if "model" in locals() or ("model_file" in locals() and dataset_file):
        if regression_option == "Upload Files" and model_file and dataset_file:
            model = load_model(model_file)
            dataset = load_dataset(dataset_file)

        if st.button("Load New Sample 🔄", key="reg_key"):
            # Load new sample and update state
            st.session_state.selected_sample = dataset.sample(1).iloc[0]
            st.session_state.slider_values = st.session_state.selected_sample.to_dict()
            st.session_state.previous_prediction = model.predict([st.session_state.selected_sample])[0]
            st.session_state.updated_prediction = st.session_state.previous_prediction

        if st.session_state.selected_sample is not None:
            sample = st.session_state.selected_sample

            with st.container():
                st.markdown('<div class="full-width">', unsafe_allow_html=True)

            # Display UI components
            layout_cols = st.columns([2, 2])

            # Left: Feature Cards
            with layout_cols[0]:
                st.subheader("Selected Features")
                display_feature_cards(sample, dataset, model)

            # Right: Predictions
            with layout_cols[1]:
                st.subheader("Predictions")
                st.session_state.updated_prediction = update_prediction(model, sample, st.session_state.slider_values)[0]
                display_predictions(st.session_state.previous_prediction, st.session_state.updated_prediction)

            # Sliders for Feature Adjustment
            st.subheader("Adjust Features")
            st.session_state.slider_values = display_sliders(sample, dataset)

        else:
            st.warning("Click 'Load New Sample' to start.")