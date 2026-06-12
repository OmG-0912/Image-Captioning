import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Import project modules
import config
import dataset

def extract_feature(image_path):
    """
    Extracts features for a single image using InceptionV3.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found at: {image_path}")

    # Load and preprocess image for InceptionV3
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Load feature extractor model
    model_cnn = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Extract feature
    feature = model_cnn.predict(img_array, verbose=0)
    return feature

def generate_caption(model, tokenizer, photo_feature, max_length):
    """
    Generates a caption for an image feature using Greedy Search.
    """
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat_idx = np.argmax(yhat)
        
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat_idx:
                word = w
                break
                
        if word is None:
            break
            
        in_text += ' ' + word
        
        if word == 'endseq':
            break
            
    # Remove startseq and endseq tokens
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def main():
    parser = argparse.ArgumentParser(description="Generate caption for a given image using the trained model.")
    parser.add_argument(
        '--image', 
        type=str, 
        default=None, 
        help="Path to the image file. If not provided, a random image from the dataset will be chosen."
    )
    args = parser.parse_args()

    # 1. Load tokenizer and metadata
    if not os.path.exists(config.TOKENIZER_PATH):
        print(f"❌ Tokenizer not found at: {config.TOKENIZER_PATH}. Run training first.")
        return
    tokenizer = dataset.load_saved_tokenizer(config.TOKENIZER_PATH)
    
    if not os.path.exists('model_metadata.pkl'):
        print("❌ model_metadata.pkl not found. Run training first.")
        return
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    max_length = metadata['max_length']

    # 2. Load trained model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"❌ Trained model not found at: {config.MODEL_SAVE_PATH}. Run training first.")
        return
    print(f"🔄 Loading model from: {config.MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print("✅ Model loaded.")

    # 3. Determine image path
    image_path = args.image
    if not image_path:
        # Pick a random image from Images folder if it exists
        if os.path.exists(config.IMAGE_DIR) and len(os.listdir(config.IMAGE_DIR)) > 0:
            import random
            all_imgs = [
                f for f in os.listdir(config.IMAGE_DIR) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if all_imgs:
                random_img = random.choice(all_imgs)
                image_path = os.path.join(config.IMAGE_DIR, random_img)
                print(f"ℹ️ No image specified. Selected random image: {image_path}")
            else:
                print(f"❌ No images found in directory: {config.IMAGE_DIR}")
                return
        else:
            print("❌ Please specify an image path using --image path/to/image.jpg")
            return

    # 4. Extract image feature and predict caption
    print(f"🔄 Extracting features for: {image_path}...")
    try:
        feature = extract_feature(image_path)
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return
        
    print("🔄 Generating caption...")
    caption = generate_caption(model, tokenizer, feature, max_length)
    
    print("\n==========================================")
    print(f"🖼️ Image: {os.path.basename(image_path)}")
    print(f"📝 Generated Caption: {caption}")
    print("==========================================\n")

if __name__ == '__main__':
    main()
