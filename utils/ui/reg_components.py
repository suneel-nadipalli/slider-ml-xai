import streamlit as st
from streamlit_vertical_slider import vertical_slider

import sys

sys.path.append("..")

from utils.helper import *

def display_feature_cards(sample, dataset, model):
    cols_per_row = 3
    num_cols = len(sample)
    rows = [sample.index[i:i + cols_per_row] for i in range(0, num_cols, cols_per_row)]

    try:
        contributions = get_coefficients(model, dataset.columns) # Replace with appropriate logic for coefficients
    except ValueError:
        contributions = compute_shap_values(model, dataset)  # Default if coefficients are unavailable

    print(contributions)

    print(sample)
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            if i < len(row):
                feature = row[i]
                value = sample[feature]
                coefficient = contributions[feature]
                value_color = "#f44336" if coefficient < 0 else "#4caf50"
                with col:
                    st.markdown(f"""
                    <div class="card">
                        <h4 style="margin: 0; color: #333;">{feature}</h4>
                        <p style="margin: 0; font-size: 20px; font-weight: bold; color: {value_color};">{value:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

def display_predictions(previous, updated):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color: #948F8FFF; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>Original Prediction</h4>
            <p style="font-size: 24px; font-weight: bold;">{previous:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: #6AC26FFF; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: black">Updated Prediction</h4>
            <p style="font-size: 24px; font-weight: bold; color: black">{updated:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

def display_sliders(sample, dataset):
    slider_values = {}
    cols = st.columns(len(sample))
    for col, feature in zip(cols, sample.index):
        
        with col:
            st.markdown(f"<div style='text-align: center; font-size: 14px;'>{feature}</div>", unsafe_allow_html=True)
            slider_values[feature] = vertical_slider(
                key=f"slider-{feature}",
                default_value=sample[feature],
                min_value=round(float(dataset[feature].min()), 2),
                max_value=round(float(dataset[feature].max()), 2),
            )
    return slider_values
