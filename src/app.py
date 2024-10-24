from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.data_loader import preprocess_data

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/trained_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data = np.array(data['data'])
    
    # Preprocess the input data
    X, _ = preprocess_data(data)
    
    # Make predictions
    predictions = model.predict(X)
    predictions = (predictions > 0.5).astype(int)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
