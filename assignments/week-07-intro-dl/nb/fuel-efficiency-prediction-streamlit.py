import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
import shap
from PIL import Image

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
model_results_df = pd.read_csv("../dat/model_results.csv")
train_df = pd.read_csv("../dat/train.csv")
tpot_shap = np.load('../dat/tpot_shap.npy')

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    # YOUR CODE GOES HERE!
    models = ["Linear Model", "DNN","DNN_Reloaded","DNN - Leaky ReLU","DNN - Leaky ReLU - Regularizer","TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    # YOUR CODE GOES HERE!
    st.dataframe(model_results_df)

    # Columns for side-by-side model comparison
    # col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    # with col1:
    #    st.header(model1_select)

        # YOUR CODE GOES HERE!


    # Build confusion matrix for second model
    #with col2:
    #    st.header(model2_select)

        # YOUR CODE GOES HERE!


with tab3: 
    # YOUR CODE GOES HERE!
        # Use columns to separate visualizations for models
        # Include plots for local and global explanability!
     
    #st.header(model1_select)
    
    #st.header(model2_select)

    st.subheader('Model Interpretability - TPOT Extra Trees Regressor') 
    fig = shap.summary_plot(tpot_shap)
    st.pyplot(fig) 
    st.write(""" In this chart we see the explainability for TPOTs top model, the Extra Trees Regressor. 
    To decode the features use this: (0, 'Cylinders'), (1, 'Displacement'), (2, 'Horsepower'), (3, 'Weight'),
     (4, 'Acceleration'), (5, 'Model Year'), (6, 'Europe'), (7, 'Japan'), (8, 'USA') 
""")
    
