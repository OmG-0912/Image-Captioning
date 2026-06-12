import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config

def clean_caption(caption):
    """
    Cleans a single caption: lowercase, removes punctuation/numbers, extra spaces.
    """
    caption = caption.lower()
    # Remove all characters except a-z and space
    caption = re.sub(r'[^a-zA-Z ]', '', caption)
    # Remove extra whitespaces
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption

def load_captions(filename):
    """
    Loads captions from a file, cleans them, and returns a dictionary of
    image_name -> list of captions wrapped with startseq and endseq.
    """
    captions_dict = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Captions file not found at: {filename}")
        
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2:
                continue
            img, caption = parts
            cleaned = clean_caption(caption)
            # Add start and end tokens
            caption_with_tokens = f"startseq {cleaned} endseq"
            captions_dict.setdefault(img, []).append(caption_with_tokens)
            
    return captions_dict

def create_tokenizer(captions_dict, save_path=None):
    """
    Fits a Tokenizer on all captions and optionally saves it.
    """
    all_captions = [cap for caps in captions_dict.values() for cap in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    # Save the tokenizer for inference/evaluation
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer, f)
            
    return tokenizer

def load_saved_tokenizer(save_path):
    """
    Loads a saved tokenizer from file.
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def create_sequences(tokenizer, max_length, captions_list, photo_feature):
    """
    Creates sequences of inputs and outputs for training from captions of one image.
    Returns:
        X1: list of image features
        X2: list of input text sequences (padded)
        y: list of target word integer indices
    """
    X1, X2, y = [], [], []
    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            # Pad the input sequence to maximum length
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            X1.append(photo_feature)
            X2.append(in_seq)
            y.append(out_seq)
    return X1, X2, y

def get_data_pipeline(captions_dict, features, tokenizer, max_length, batch_size, shuffle=True):
    """
    Creates an optimized tf.data.Dataset pipeline.
    """
    # Create lists of valid image names that have features
    valid_imgs = [img for img in captions_dict.keys() if img in features]
    
    def generator():
        # Shuffle image list each epoch if training
        img_list = list(valid_imgs)
        if shuffle:
            np.random.shuffle(img_list)
            
        for img_name in img_list:
            photo_feature = features[img_name][0]  # Shape: (IMAGE_FEATURE_DIM,)
            captions = captions_dict[img_name]
            X1, X2, y = create_sequences(tokenizer, max_length, captions, photo_feature)
            for i in range(len(X1)):
                yield (X1[i], X2[i]), y[i]
                
    output_signature = (
        (
            tf.TensorSpec(shape=(config.IMAGE_FEATURE_DIM,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Batch and prefetch the dataset for high throughput
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
