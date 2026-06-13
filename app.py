import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import project modules
import config
import dataset
import predict

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Global cache for the ML models to ensure fast inference responses
model_cache = {
    'model': None,
    'tokenizer': None,
    'max_length': None,
    'model_cnn': None
}

def load_prediction_model():
    """
    Lazy-loads the model, tokenizer, and feature extractor into memory.
    """
    # Load CNN feature extractor if not cached
    if model_cache['model_cnn'] is None:
        print("🔄 Loading InceptionV3 feature extractor into server cache...")
        model_cache['model_cnn'] = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        print("✅ InceptionV3 model loaded and cached.")

    if model_cache['model'] is not None:
        return (
            model_cache['model'], 
            model_cache['tokenizer'], 
            model_cache['max_length'],
            model_cache['model_cnn']
        )
        
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_SAVE_PATH}. "
            "Please run 'train.py' or 'run_test_suite.py' first to train the model."
        )
        
    if not os.path.exists(config.TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at {config.TOKENIZER_PATH}. "
            "Please train the model first."
        )
        
    if not os.path.exists('model_metadata.pkl'):
        raise FileNotFoundError(
            "model_metadata.pkl not found. Please train the model first."
        )

    print("🔄 Loading Keras model and tokenizer into server cache...")
    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        max_length = metadata['max_length']
        
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print("✅ Model loaded successfully and cached.")
    
    model_cache['model'] = model
    model_cache['tokenizer'] = tokenizer
    model_cache['max_length'] = max_length
    
    return model, tokenizer, max_length, model_cache['model_cnn']

@app.route('/')
def index():
    """
    Serves the main frontend page.
    """
    return app.send_static_file('index.html')

@app.route('/caption', methods=['POST'])
def get_caption():
    """
    Receives an image via POST, extracts features, and returns the generated caption.
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Selected file is empty'}), 400

    try:
        # Create temp folder inside workspace for uploads
        temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Load ML components
        model, tokenizer, max_length, model_cnn = load_prediction_model()
        
        # Extract features and predict
        feature = predict.extract_feature(temp_path, model_cnn=model_cnn)
        caption = predict.generate_caption(model, tokenizer, feature, max_length)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            'success': True,
            'caption': caption
        })
        
    except FileNotFoundError as e:
        return jsonify({
            'success': False, 
            'error': str(e) + " Run the training script first to save the model."
        }), 500
    except Exception as e:
        # Generic fallback
        return jsonify({
            'success': False, 
            'error': f"Failed to generate caption: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=False)
