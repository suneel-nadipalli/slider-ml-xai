import sys
sys.path.append("..")
from utils.helper import *
from utils.ui.reg_components import *

def reset_regression_state():
    """Resets regression-related session state variables."""
    st.session_state.reg_selected_sample = None
    st.session_state.reg_slider_values = None
    st.session_state.reg_previous_prediction = None
    st.session_state.reg_updated_prediction = None

def display_reg():
    st.header("Regression")
    
    # Initialize session state variables for regression if not present
    if "reg_selected_sample" not in st.session_state:
        reset_regression_state()

    # Choose regression setup
    regression_option = st.radio(
        "Choose a regression setup:",
        ("Upload Files", "Use Preloaded Models/Datasets"),
        horizontal=True,
    )

    model, dataset = None, None

    if regression_option == "Upload Files":
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Upload your model (.pkl):", type=["pkl"], key="reg_model_file")
        with col2:
            dataset_file = st.file_uploader("Upload your dataset (.csv):", type=["csv"], key="reg_dataset_file")

        if model_file and dataset_file:
            if st.session_state.last_regression_model != model_file or st.session_state.last_regression_dataset != dataset_file:
                st.session_state.last_regression_model = model_file
                st.session_state.last_regression_dataset = dataset_file
                reset_regression_state()

            model = load_model(model_file)
            dataset = load_dataset(dataset_file)

    else:
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

        if st.session_state.last_regression_model != model_path or st.session_state.last_regression_dataset != dataset_path:
            st.session_state.last_regression_model = model_path
            st.session_state.last_regression_dataset = dataset_path
            reset_regression_state()

        model = load_model(model_path)
        dataset = load_dataset(dataset_path)

    # Load new sample and update state
    if model and dataset is not None:
        if st.button("Load New Sample 🔄", key="reg_load_sample"):
            st.session_state.reg_selected_sample = dataset.sample(1).iloc[0]
            st.session_state.reg_slider_values = st.session_state.reg_selected_sample.to_dict()
            st.session_state.reg_previous_prediction = model.predict([st.session_state.reg_selected_sample])[0]
            st.session_state.reg_updated_prediction = st.session_state.reg_previous_prediction

        if st.session_state.reg_selected_sample is not None:
            sample = st.session_state.reg_selected_sample

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
                st.session_state.reg_updated_prediction = update_prediction(
                    model, sample, st.session_state.reg_slider_values
                )[0]
                display_predictions(st.session_state.reg_previous_prediction, st.session_state.reg_updated_prediction)

            # Sliders for Feature Adjustment
            st.subheader("Adjust Features")
            st.session_state.reg_slider_values = display_sliders(sample, dataset)

        else:
            st.warning("Click 'Load New Sample' to start.")
