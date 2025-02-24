
import streamlit as st
import numpy as np
import pickle

def load_model():
    try:
        with open(r'C:\Users\jacok\Downloads\iris_model.pkl', 'rb') as file:
            model = pickle.load(file)# Loading the model from the pickle file
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
    
    # Adding title with custom styling
    st.markdown("""
        <style>
        .big-title {
            font-size: 50px;
            color: #3366cc;
            text-align: center;
        }
        </style>
        <h1 class='big-title'>Iris Flower Classifier</h1>
    """, unsafe_allow_html=True)
    
    # Loading the model
    model = load_model()
    
    if model is None:
        st.warning("Please make sure 'iris_model.pkl' is in the same directory as this app")
        return
    
    # Creating two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Flower Measurements")
        # Creating a form for input
        with st.form("prediction_form"):
            # Two columns for inputs
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                sepal_length = st.number_input('Sepal Length (cm)', 
                                             min_value=0.0, 
                                             max_value=10.0, 
                                             value=5.0,
                                             step=0.1)
                sepal_width = st.number_input('Sepal Width (cm)', 
                                            min_value=0.0, 
                                            max_value=10.0, 
                                            value=3.5,
                                            step=0.1)
            
            with input_col2:
                petal_length = st.number_input('Petal Length (cm)', 
                                             min_value=0.0, 
                                             max_value=10.0, 
                                             value=1.4,
                                             step=0.1)
                petal_width = st.number_input('Petal Width (cm)', 
                                            min_value=0.0, 
                                            max_value=10.0, 
                                            value=0.2,
                                            step=0.1)
            
            predict_button = st.form_submit_button("Predict Iris Type")
    
        if predict_button:
            # Preparing the  input data
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Making a prediction
            prediction = model.predict(input_data)
            iris_types = ['Setosa', 'Versicolor', 'Virginica']
            predicted_iris = iris_types[prediction[0]]
            
            # Displaying prediction with custom styling
            st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
                    <h3 style='color: #3366cc; text-align: center;'>Prediction Result</h3>
                    <p style='font-size: 24px; text-align: center;'>
                        This iris flower is predicted to be: 
                        <strong style='color: #3366cc;'>{predicted_iris}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Displaying input summary
        st.subheader("Input Summary")
        summary_data = {
            'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        }
        st.table(summary_data)
        
        # Adding information about iris types
        st.subheader("Iris Types Information")
        st.markdown("""
        **Setosa**
        - Usually the smallest of the three
        - Known for its distinctive appearance
        
        **Versicolor**
        - Medium-sized iris
        - Shows color variations
        
        **Virginica**
        - Typically the largest
        - Known for elegant flowers
        """)

if __name__ == '__main__':
    main()