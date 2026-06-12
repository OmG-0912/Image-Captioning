import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import project modules
import config
import dataset
import model

# Set random seed for reproducibility
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

def extract_features_if_needed():
    """
    Checks if features.pkl exists. If not, extracts features for all images
    in config.IMAGE_DIR using InceptionV3 and saves them to features.pkl.
    """
    if os.path.exists(config.FEATURES_PATH):
        print(f"ℹ️ Pre-extracted features found at: {config.FEATURES_PATH}")
        with open(config.FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        print(f"✅ Loaded features for {len(features)} images.")
        return features

    print(f"⚠️ Features file {config.FEATURES_PATH} not found.")
    print(f"🔄 Starting feature extraction using InceptionV3 from '{config.IMAGE_DIR}'...")
    
    if not os.path.exists(config.IMAGE_DIR):
        raise FileNotFoundError(
            f"❌ Image directory '{config.IMAGE_DIR}' not found. "
            f"Please download the Flickr8k images and place them in this folder."
        )

    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # Load pre-trained InceptionV3
    model_cnn = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    
    img_names = [
        f for f in os.listdir(config.IMAGE_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    total_imgs = len(img_names)
    print(f"Found {total_imgs} images. Starting batch extraction...")
    
    for i, img_name in enumerate(img_names):
        path = os.path.join(config.IMAGE_DIR, img_name)
        try:
            image = load_img(path, target_size=(299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            feature = model_cnn.predict(image, verbose=0)
            features[img_name] = feature
            
            if (i + 1) % 100 == 0 or (i + 1) == total_imgs:
                print(f"Processed {i + 1}/{total_imgs} images...")
        except Exception as e:
            print(f"⚠️ Failed to process {img_name}: {e}")

    # Save to file
    with open(config.FEATURES_PATH, 'wb') as f:
        pickle.dump(features, f)
    print(f"✅ Extracted & saved features for {len(features)} images to: {config.FEATURES_PATH}")
    return features

def train():
    # 1. Extract or load features
    try:
        features = extract_features_if_needed()
    except Exception as e:
        print(e)
        return

    # 2. Load and clean captions
    print("🔄 Loading and cleaning captions...")
    try:
        captions_dict = dataset.load_captions(config.CAPTIONS_PATH)
    except FileNotFoundError as e:
        print(e)
        print("❌ Please ensure captions.txt exists in the project root directory.")
        return
    print(f"✅ Captions loaded: {len(captions_dict)} images cataloged.")

    # Filter out images from captions dictionary that don't have extracted features
    valid_keys = [k for k in captions_dict.keys() if k in features]
    print(f"📊 Valid images with both features and captions: {len(valid_keys)}")

    # 3. Train-Test Split (Images level split to prevent data leakage)
    train_keys, val_keys = train_test_split(
        valid_keys, 
        test_size=config.VAL_SPLIT, 
        random_state=config.RANDOM_STATE
    )
    print(f"📈 Split details: Train={len(train_keys)} images | Val={len(val_keys)} images")

    # Re-build train and val captions dicts
    train_captions = {k: captions_dict[k] for k in train_keys}
    val_captions = {k: captions_dict[k] for k in val_keys}

    # 4. Tokenization (Only fit on training set to prevent vocabulary leakage)
    print("🔄 Creating and fitting tokenizer on training data...")
    tokenizer = dataset.create_tokenizer(train_captions, save_path=config.TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Calculate max length dynamically on train set
    all_train_caps = [cap for caps in train_captions.values() for cap in caps]
    max_length = max(len(cap.split()) for cap in all_train_caps)
    
    print(f"✅ Tokenizer saved to: {config.TOKENIZER_PATH}")
    print(f"📊 Vocabulary Size: {vocab_size} | Maximum Caption Length: {max_length}")

    # Save metadata for eval and inference config values updating
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'max_length': max_length}, f)

    # 5. Build tf.data.Dataset pipelines
    print("🔄 Preparing optimized tf.data.Dataset pipelines...")
    train_dataset = dataset.get_data_pipeline(
        train_captions, 
        features, 
        tokenizer, 
        max_length, 
        config.BATCH_SIZE, 
        shuffle=True
    )
    
    val_dataset = dataset.get_data_pipeline(
        val_captions, 
        features, 
        tokenizer, 
        max_length, 
        config.BATCH_SIZE, 
        shuffle=False
    )

    # Calculate steps per epoch
    def count_sequences(caps_dict):
        seq_count = 0
        for img_name, caps in caps_dict.items():
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                seq_count += len(seq) - 1
        return seq_count

    train_seqs = count_sequences(train_captions)
    val_seqs = count_sequences(val_captions)
    steps_per_epoch = train_seqs // config.BATCH_SIZE
    validation_steps = val_seqs // config.BATCH_SIZE
    
    print(f"📊 Training sequences: {train_seqs} (Steps per epoch: {steps_per_epoch})")
    print(f"📊 Validation sequences: {val_seqs} (Validation steps: {validation_steps})")

    # 6. Instantiate Model
    print("🔄 Building the model...")
    img_captioner = model.build_merge_model(vocab_size, max_length)
    img_captioner.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    )
    img_captioner.summary()

    # 7. Callbacks Setup
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=0,
            write_graph=True
        )
    ]

    # 8. Start Training
    print("🚀 Starting training...")
    history = img_captioner.fit(
        train_dataset,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    print("🎉 Training finished!")
    print(f"💾 Best model weights saved to: {config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
