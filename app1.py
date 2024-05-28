import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load model
with open('House_Predict.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize scaler
scaler = MinMaxScaler()

# Load data
data = pd.read_csv('test.csv')  # Thay 'your_data.csv' bằng tên tệp dữ liệu của bạn
# Fit scaler to your data
scaler.fit(data)

# Function to preprocess input data
def preprocess_input(data):
    return scaler.transform(data)

# Function to make predictions


# Main function
def main():
    # Set page style
    st.markdown("<body style='color:#E2E0D9;'></body>", unsafe_allow_html=True)

    # Set page title and description
    st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>A multi-step process is used to estimate the range of house prices based on your selection. The modeling process is done using the data found on Kaggle(link at left bottom corner of page)</h5>", unsafe_allow_html=True)

    # Define input variables and their descriptions
    name_list = ['MSSubClass', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF', 'FstFlrSF', 'SndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MoSold', 'YrSold']
    description_list = ['What is the building class?', 'What is the Overall material and finish quality?', 'In which year was the Original construction date?', 'In which year was it remodelled?', 'What is the Unfinished square feet of basement area?', 'What is the Total square feet of basement area?', 'What is the First Floor square feet?', 'What is the Second floor square feet?', 'What is the Above grade (ground) living area square feet?', 'What is the number of full bathrooms?', 'What is the number of Half baths?', 'What is the number of Total rooms above grade (does not include bathrooms)?', 'What is the number of fireplaces?', 'What is the garage capacity in car sizes?', 'What is the size of garage in square feet?', 'In which month was it sold?', 'In which year was it sold?']
    min_list = [20.0, 1.0, 1872.0, 1950.0, 0.0, 0.0, 334.0, 0.0, 334.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2006.0]
    max_list = [190.0, 10.0, 2010.0, 2010.0, 2336.0, 6110.0, 4692.0, 2065.0, 5642.0, 3.0, 2.0, 14.0, 3.0, 4.0, 1418.0, 12.0, 2030.0]

    # Display sidebar
    with st.sidebar:
        for i in range(len(name_list)):
            variable_name = name_list[i]
            globals()[variable_name] = st.slider(description_list[i], min_value=int(min_list[i]), max_value=int(max_list[i]), step=1)
        st.write("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)")

    # Check if the 'Predict' button is clicked
    if st.button('Predict', key='predict_button'):
        # Create a dataframe with the input data
        input_data = pd.DataFrame({
            'MSSubClass': [MSSubClass],
            'OverallQual': [OverallQual],
            'YearBuilt': [YearBuilt],
            'YearRemodAdd': [YearRemodAdd],
            'BsmtUnfSF': [BsmtUnfSF],
            'TotalBsmtSF': [TotalBsmtSF],
            'FstFlrSF': [FstFlrSF],
            'SndFlrSF': [SndFlrSF],
            'GrLivArea': [GrLivArea],
            'FullBath': [FullBath],
            'HalfBath': [HalfBath],
            'TotRmsAbvGrd': [TotRmsAbvGrd],
            'Fireplaces': [Fireplaces],
            'GarageCars': [GarageCars],
            'GarageArea': [GarageArea],
            'MoSold': [MoSold],
            'YrSold': [YrSold]
        })

        # Preprocess the input data
        input_data = pd.get_dummies(input_data, drop_first=True)
        scaled_data = preprocess_input(input_data)

        # Make predictions
        predictions = predict(scaled_data)

        # Display the predictions
        st.write('Predicted house prices:', predictions)

if __name__ == '__main__':
    main()
