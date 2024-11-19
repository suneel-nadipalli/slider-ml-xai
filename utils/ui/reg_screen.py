import sys
sys.path.append("..")
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

    # Track the previously selected model/dataset to detect changes
    if "last_regression_model" not in st.session_state:
        st.session_state.last_regression_model = None
    if "last_regression_dataset" not in st.session_state:
        st.session_state.last_regression_dataset = None

    model, dataset = None, None

    if regression_option == "Upload Files":
        # File upload
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Upload your model (.pkl):", type=["pkl"])
        with col2:
            dataset_file = st.file_uploader("Upload your dataset (.csv):", type=["csv"])

        if model_file and dataset_file:
            # Detect changes and reset state
            if st.session_state.last_regression_model != model_file or st.session_state.last_regression_dataset != dataset_file:
                st.session_state.last_regression_model = model_file
                st.session_state.last_regression_dataset = dataset_file
                reset_regression_state()

            model = load_model(model_file)
            dataset = load_dataset(dataset_file)

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
            model_path, dataset_path = "models/reg_house_price.pkl", "data/reg_house_price_test.csv"
        elif preloaded_option == "Decision Tree Regressor + Auto MPG Dataset":
            model_path, dataset_path = "models/reg_tree_miles.pkl", "data/reg_tree_miles_test.csv"

        # Detect changes and reset state
        if st.session_state.last_regression_model != model_path or st.session_state.last_regression_dataset != dataset_path:
            st.session_state.last_regression_model = model_path
            st.session_state.last_regression_dataset = dataset_path
            reset_regression_state()

        model = load_model(model_path)
        dataset = load_dataset(dataset_path)

    # If files are uploaded or preloaded models are selected
    if model and dataset is not None:
        if st.button("Load New Sample 🔄", key="reg_key"):
            # Load new sample and update state
            st.session_state.selected_sample_reg = dataset.sample(1).iloc[0]
            st.session_state.slider_values_reg = st.session_state.selected_sample_reg.to_dict()
            st.session_state.previous_prediction_reg = model.predict([st.session_state.selected_sample_reg])[0]
            st.session_state.updated_prediction_reg = st.session_state.previous_prediction_reg

        if st.session_state.selected_sample_reg is not None:
            sample = st.session_state.selected_sample_reg

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
                st.session_state.updated_prediction_reg = update_prediction(
                    model, sample, st.session_state.slider_values_reg
                )[0]
                display_predictions(st.session_state.previous_prediction_reg, st.session_state.updated_prediction_reg)

            # Sliders for Feature Adjustment
            st.subheader("Adjust Features")
            st.session_state.slider_values_reg = display_sliders(sample, dataset)

        else:
            st.warning("Click 'Load New Sample' to start.")

def reset_regression_state():
    """Resets regression-related session state variables."""
    st.session_state.selected_sample_reg = None
    st.session_state.slider_values_reg = None
    st.session_state.previous_prediction_reg = None
    st.session_state.updated_prediction_reg = None
