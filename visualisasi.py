import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from laptop_harga import laptop_harga
st.set_option('deprecation.showPyplotGlobalUse', False)
# Load the dataset
df = pd.read_csv('Laptop_price.csv')

def explore_data():
    # Display total rows and columns
    st.write("Total Rows and Columns:", df.shape)

    # Display data overview
    st.write("Data Overview:", df.head())

    # Check for missing values
    st.write("Missing Values:", df.isnull().sum())

def visualize_brand_distribution():
    # Visualize brand distribution
    brand_counts = df['Brand'].value_counts()
    st.bar_chart(brand_counts)

def visualize_correlation():
    # Visualize correlation matrix
    df_corr = df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']]
    corr = df_corr.corr()
    st.write("Correlation Matrix:", corr)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()

def visualize_price_distribution():
    # Visualize price distribution
    sns.histplot(df['Price'], kde=True)
    st.pyplot()

def visualize_price_by_brand():
    # Visualize price distribution by brand
    fig, ax = plt.subplots(figsize=(12, 6))
    for brand in df['Brand'].unique():
        subset_data = df[df['Brand'] == brand]
        sns.histplot(subset_data['Price'], ax=ax, label=brand, alpha=0.5)
    ax.set_title('Pricing Patterns of Brands')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot()

def train_model():
    # Prepare data for modeling
    df_encoded = pd.get_dummies(df)
    X, y = df_encoded.drop(['Price'], axis=1), df_encoded['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)

    # Train Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    return model_rf, model_lr

def predict_price(model, user_input):
    predicted_price = model.predict(user_input)
    return predicted_price

def main():
    st.title('Laptop Price Prediction Dashboard')

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Explore Data", "Visualize Data", "Train Model", "Predict Price"])

    if page == "Explore Data":
        explore_data()

    elif page == "Visualize Data":
        visualization_option = st.selectbox("Choose a visualization", ["Brand Distribution", "Correlation Matrix", "Price Distribution", "Price by Brand"])
        if visualization_option == "Brand Distribution":
            visualize_brand_distribution()
        elif visualization_option == "Correlation Matrix":
            visualize_correlation()
        elif visualization_option == "Price Distribution":
            visualize_price_distribution()
        elif visualization_option == "Price by Brand":
            visualize_price_by_brand()

    elif page == "Train Model":
        model_rf, model_lr = train_model()
        st.write("Models Trained Successfully!")

    elif page == "Predict Price":
        laptop_harga()

if __name__ == "__main__":
    main()
