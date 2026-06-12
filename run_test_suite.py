import os
import shutil
import numpy as np
import pickle
from PIL import Image

# Import configurations
import config

def generate_mock_data():
    """
    Programmatically generates a miniature toy dataset:
    - 4 random color images inside config.IMAGE_DIR
    - A captions.txt file containing 5 captions for each image
    """
    print("🔄 Generating mock dataset...")
    # Create Images directory
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    
    # Save 4 mock images
    mock_images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    for img_name in mock_images:
        img_path = os.path.join(config.IMAGE_DIR, img_name)
        # Create a 299x299 RGB image with random colors
        arr = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_path)
    
    # Save captions.txt
    captions = [
        "image,caption",
        "img1.jpg,a dog runs in the green grass",
        "img1.jpg,a dog playing in a field",
        "img1.jpg,a brown dog runs outdoors",
        "img1.jpg,the dog is running outside",
        "img1.jpg,a dog is sprinting on the lawn",
        "img2.jpg,a white cat sits on a red sofa",
        "img2.jpg,a fluffy cat rests on the couch",
        "img2.jpg,the cat is lounging indoors",
        "img2.jpg,a feline sits comfortably on furniture",
        "img2.jpg,a sleeping cat on a red cushion",
        "img3.jpg,a man rides a bicycle on the street",
        "img3.jpg,someone cycling down the road",
        "img3.jpg,a cyclist on a black bike",
        "img3.jpg,the person is riding a bicycle",
        "img3.jpg,a bike rider on a paved path",
        "img4.jpg,two children play in the sand",
        "img4.jpg,kids playing at the beach",
        "img4.jpg,young children building a sandcastle",
        "img4.jpg,a boy and girl play on the shore",
        "img4.jpg,children having fun in the sand"
    ]
    
    with open(config.CAPTIONS_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(captions))
        
    print("✅ Mock dataset created successfully.")

def run_pipeline():
    """
    Alters configuration values for speed and runs training, evaluation, and prediction.
    """
    # Overwrite configurations temporarily for hyper-fast execution
    config.EPOCHS = 2
    config.BATCH_SIZE = 4
    config.VAL_SPLIT = 0.25
    
    print("\n==========================================")
    print("🔄 Running Train Pipeline...")
    import train
    train.train()
    print("✅ Train Pipeline complete.")
    
    print("\n==========================================")
    print("🔄 Running Evaluation Pipeline (BLEU)...")
    import eval
    eval.evaluate_model()
    print("✅ Evaluation Pipeline complete.")
    
    print("\n==========================================")
    print("🔄 Running Inference Test on dummy image...")
    import predict
    test_img = os.path.join(config.IMAGE_DIR, "img1.jpg")
    
    # Load model cache to do inference check
    import tensorflow as tf
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        
    feature = predict.extract_feature(test_img)
    caption = predict.generate_caption(model, tokenizer, feature, metadata['max_length'])
    print(f"🖼️ Test Image: {test_img}")
    print(f"📝 Generated Caption: {caption}")
    print("✅ Inference Test complete.")
    print("==========================================\n")

def main():
    # 1. Clean up old test data if present
    for path in [
        config.IMAGE_DIR, 
        config.CAPTIONS_PATH, 
        config.FEATURES_PATH, 
        config.MODEL_SAVE_PATH, 
        config.TOKENIZER_PATH, 
        'model_metadata.pkl',
        'temp_uploads'
    ]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
    # 2. Generate new mock data
    generate_mock_data()
    
    # 3. Run pipelines
    try:
        run_pipeline()
        print("🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("🤖 Model weights and tokenizer are now ready for the Flask server.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
