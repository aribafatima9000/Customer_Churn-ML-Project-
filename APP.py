from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("model_scaler.pkl")
    print("[SUCCESS] Model loaded successfully!")
    print("[SUCCESS] Scaler loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None
    scaler = None

# Define categorical columns for one-hot encoding
CATEGORICAL_COLS = ['country', 'city', 'customer_segment', 'signup_channel', 
                     'contract_type', 'payment_method', 'complaint_type']

# Define possible values for one-hot encoding (must match training data)
ONEHOT_MAPPINGS = {
    'country': ['USA', 'Canada', 'UK', 'Australia', 'Germany'],
    'city': ['New York', 'Los Angeles', 'London', 'Sydney', 'Toronto', 'Berlin'],
    'customer_segment': ['SME', 'Individual', 'Enterprise'],
    'signup_channel': ['Website', 'Mobile App', 'Sales Team', 'Referral'],
    'contract_type': ['Month-to-Month', 'Annual', 'Multi-Year'],
    'payment_method': ['Credit Card', 'Bank Transfer', 'Check'],
    'complaint_type': ['Billing', 'Technical', 'Service', 'Other']
}

def encode_features(form_data):
    """
    Encode raw customer features to match training data format.
    Returns a numpy array of 47 features ready for model prediction.
    """
    
    # Step 1: Label Encode binary categorical features
    gender = 1 if form_data.get('gender') == 'Male' else 0
    survey_response = 1 if form_data.get('survey_response') == 'Yes' else 0
    discount_applied = 1 if form_data.get('discount_applied') == 'Yes' else 0
    price_increase_last_3m = 1 if form_data.get('price_increase_last_3m') == 'Yes' else 0
    
    # Step 2: Get numeric features
    tenure = float(form_data.get('tenure', 0))
    csat_score = float(form_data.get('csat_score', 0))
    
    # Step 3: One-hot encode categorical features using drop_first=True logic
    onehot_features = []
    
    for col in CATEGORICAL_COLS:
        col_value = form_data.get(col, '')
        possible_values = ONEHOT_MAPPINGS.get(col, [])
        
        # drop_first=True means we skip the first category
        for i, category in enumerate(possible_values[1:], 1):  # Skip first category
            if col_value == category:
                onehot_features.append(1)
            else:
                onehot_features.append(0)
    
    # Step 4: Combine all features in the correct order
    # Order: label_encoded features + numeric + onehot
    all_features = [
        gender,
        survey_response,
        discount_applied,
        price_increase_last_3m,
        tenure,
        csat_score,
    ] + onehot_features
    
    return np.array(all_features, dtype=float)

def get_feature_importance_plot(encoded_features):
    """Create a simple feature importance visualization"""
    try:
        # Get coefficients from logistic regression
        coefficients = model.coef_[0]
        
        # Get top 10 important features
        importance = np.abs(coefficients)
        top_indices = np.argsort(importance)[-10:][::-1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = [f'F{i+1}' for i in range(len(coefficients))]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = [importance[i] for i in top_indices]
        
        colors = ['#f5576c' if coefficients[i] > 0 else '#4facfe' for i in top_indices]
        ax.barh(top_features, top_importance, color=colors)
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Top 10 Feature Importance (Logistic Regression)', fontweight='bold')
        ax.invert_yaxis()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    except:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check server logs.",
                                 show_result=True)
        
        # Extract original features from form
        form_data = request.form.to_dict()
        
        # Encode features automatically
        encoded_features = encode_features(form_data)
        
        # Reshape for model (expecting 2D array)
        encoded_features_reshaped = encoded_features.reshape(1, -1)
        
        # Scale features using trained scaler
        final_features_scaled = scaler.transform(encoded_features_reshaped)
        
        # Get prediction probability
        prediction_proba = model.predict_proba(final_features_scaled)[0][1]
        
        # Apply threshold of 0.3 for churn prediction
        prediction = 1 if prediction_proba >= 0.3 else 0
        
        # Format result
        if prediction == 1:
            result_text = "Customer Will Churn [HIGH RISK]"
            confidence = f"{prediction_proba * 100:.1f}%"
            risk_level = "HIGH"
            action = "Recommend: Immediate retention offer and customer support intervention"
        else:
            result_text = "Customer Will Stay [LOW RISK]"
            confidence = f"{(1 - prediction_proba) * 100:.1f}%"
            risk_level = "LOW"
            action = "Recommend: Maintain service quality and continue engagement"
        
        # Get feature importance plot
        plot_url = get_feature_importance_plot(encoded_features)
        
        return render_template('index.html',
                             prediction_text=result_text,
                             confidence=confidence,
                             risk_level=risk_level,
                             action=action,
                             plot_url=plot_url,
                             show_result=True)
    
    except Exception as e:
        return render_template('index.html',
                             error=f"Error occurred: {str(e)}",
                             show_result=True)

if __name__ == "__main__":
    app.run(debug=True)
