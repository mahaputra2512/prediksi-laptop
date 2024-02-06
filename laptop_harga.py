import streamlit as st
import joblib
import numpy as np

def laptop_harga() :
    model_path = 'linear_regression_model.h5'  # Update the path accordingly
    lin_reg = joblib.load(model_path)

    # Set the page configuration for a cyber theme
    

    # Function to get user input for laptop features
    def get_user_input():
        st.subheader("Enter the laptop features:")
        processor_speed = st.number_input("Processor Speed", min_value=0.0, step=0.1)
        ram_size = st.number_input("RAM Size (GB)", min_value=1.0, step=1.0)
        storage_capacity = st.number_input("Storage Capacity (GB)", min_value=1.0, step=1.0)
        screen_size = st.number_input("Screen Size", min_value=10.0, step=0.1)
        weight = st.number_input("Weight", min_value=0.1, step=0.1)

        # Dropdown for selecting laptop brand
        laptop_brand = st.selectbox("Select Laptop Brand", ["acer", "asus", "dell", "hp", "lenovo"])

        # Convert brand to 1 if selected, 0 otherwise
        acer = 1 if laptop_brand == "acer" else 0
        asus = 1 if laptop_brand == "asus" else 0
        dell = 1 if laptop_brand == "dell" else 0
        hp = 1 if laptop_brand == "hp" else 0
        lenovo = 1 if laptop_brand == "lenovo" else 0

        return np.array([[processor_speed, ram_size, storage_capacity, screen_size, weight, acer, asus, dell, hp, lenovo]])

    # Get user input
    user_input = get_user_input()

    # Make a prediction using the loaded model
    predicted_price = lin_reg.predict(user_input)

    # Apply cyber theme with some custom CSS
    st.markdown(
        """
        <style>
            body {
                background-color: #111111;
                color: #00FF41;
            }
            .st-ba {
                background-color: #001F3F;
                color: #00FF41;
            }
            .st-at,
            .st-ae {
                color: #00FF41;
            }
            .st-ae:hover {
                background-color: #001F3F;
            }
            .css-19ih76x {
                color: #00FF41;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the predicted price
    st.subheader(f"Predicted Laptop Price: ${predicted_price[0]:.2f}")
