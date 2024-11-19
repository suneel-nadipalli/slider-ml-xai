import time
import streamlit as st
from streamlit_vertical_slider import vertical_slider
# import sys

# sys.path.append("..")

from utils.helper import *

from utils.ui.reg_components import *
from utils.ui.reg_screen import *

from utils.ui.clf_screen import *
from utils.ui.clf_components import *

st.set_page_config(layout="wide")

# CSS for styling
st.markdown("""
<style>
    .full-width {
        margin: 20px auto; /* Add vertical spacing between rows */
        width: 95%; /* Occupy 95% of the page width */
    }
    .card {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #ddd;
        margin-bottom: 20px; /* Add spacing between cards */
    }
</style>
""", unsafe_allow_html=True)

st.title("""
Welcome to Slider ML! 🎉
         
A tool to help interactively explore and understand machine learning models, and the factors that effect their predictions.
         
Choose a task from the tabs below to get started.
         
For more info on how to use the tool, check out the sidebar on the left.
"""
)

st.sidebar.title("Welcome!")
st.sidebar.info("""

### **How to Use:**
1. **Choose a Task**: Use the **Regression** or **Classification** tab.
2. **Upload or Select Data**: Upload your own `.pkl` model and `.csv` dataset, or use a preloaded option.
3. **Load a Sample**: Click **Load New Sample** to select and analyze a data point.
4. **Adjust Features**: Use sliders to modify feature values and see how predictions change.
5. **Understand Predictions**: 
   - For **classification**, view real-time class probabilities with a split bar.
   - For **regression**, see dynamic updates to predictions.

### **Why Use This Tool?**
- Gain insights into model behavior.
- Experiment with scenarios.
- Build confidence in ML predictions.

Happy exploring!
""")

# Initialize session state
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "slider_values" not in st.session_state:
    st.session_state.slider_values = None
if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None
if "updated_prediction" not in st.session_state:
    st.session_state.updated_prediction = None
if "previous_proba" not in st.session_state:
    st.session_state.previous_proba = None
if "updated_proba" not in st.session_state:
    st.session_state.updated_proba = None

# Tabs for Regression and Classification
tab1, tab2 = st.tabs(["Regression", "Classification"])

# --- REGRESSION TAB ---
with tab1:
    display_reg()

# --- CLASSIFICATION TAB ---
with tab2:
    display_clf()