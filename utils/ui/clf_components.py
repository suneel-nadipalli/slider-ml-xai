import streamlit as st
from streamlit_vertical_slider import vertical_slider

def display_split_rectangle(probabilities, class_names):
    class_0_prob = probabilities[0] * 100
    class_1_prob = probabilities[1] * 100
    st.markdown(f"""
    <div style="display: flex; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; height: 50px;">
        <div style="flex: {class_0_prob}; background-color: #2196F3; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold;">
            {class_names[0]}: {class_0_prob:.1f}%
        </div>
        <div style="flex: {class_1_prob}; background-color: #f44336; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold;">
            {class_names[1]}: {class_1_prob:.1f}%
        </div>
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

def display_feature_importances(dataset, model):
    feature_importances = model.feature_importances_
    for feature, importance in zip(dataset.columns, feature_importances):
        st.write(f"{feature}: {importance:.2f}")
