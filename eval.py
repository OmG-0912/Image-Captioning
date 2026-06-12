import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

# Import project modules
import config
import dataset

def generate_caption(model, tokenizer, photo_feature, max_length):
    """
    Generates a caption for an image feature using Greedy Search.
    """
    in_text = 'startseq'
    for _ in range(max_length):
        # Integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad input sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word probabilities
        yhat = model.predict([photo_feature, sequence], verbose=0)
        # Get index with highest probability
        yhat_idx = np.argmax(yhat)
        
        # Map integer index to word
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat_idx:
                word = w
                break
                
        # End of sequence or unknown index
        if word is None:
            break
            
        # Append predicted word
        in_text += ' ' + word
        
        # End sequence check
        if word == 'endseq':
            break
            
    return in_text

def evaluate_model():
    # 1. Load tokenizer and metadata
    if not os.path.exists(config.TOKENIZER_PATH):
        raise FileNotFoundError(f"❌ Tokenizer not found at: {config.TOKENIZER_PATH}. Run training first.")
    tokenizer = dataset.load_saved_tokenizer(config.TOKENIZER_PATH)
    
    if not os.path.exists('model_metadata.pkl'):
        raise FileNotFoundError("❌ model_metadata.pkl not found. Run training first.")
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    max_length = metadata['max_length']

    # 2. Load trained model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"❌ Trained model not found at: {config.MODEL_SAVE_PATH}. Run training first.")
    print(f"🔄 Loading trained model from: {config.MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print("✅ Model loaded successfully.")

    # 3. Load features
    if not os.path.exists(config.FEATURES_PATH):
        raise FileNotFoundError(f"❌ Features file not found at: {config.FEATURES_PATH}.")
    with open(config.FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)

    # 4. Load captions and split data identically to train.py
    captions_dict = dataset.load_captions(config.CAPTIONS_PATH)
    valid_keys = [k for k in captions_dict.keys() if k in features]
    
    _, val_keys = train_test_split(
        valid_keys, 
        test_size=config.VAL_SPLIT, 
        random_state=config.RANDOM_STATE
    )
    print(f"📊 Running evaluation on {len(val_keys)} validation images...")

    # 5. Evaluate on Validation Set
    references = []
    hypotheses = []
    
    total_val = len(val_keys)
    for idx, img_name in enumerate(val_keys):
        photo_feature = features[img_name]  # Shape: (1, 2048)
        
        # Generate caption
        generated_raw = generate_caption(model, tokenizer, photo_feature, max_length)
        
        # Remove startseq and endseq tokens, split into list of words
        generated = [
            w for w in generated_raw.split() 
            if w not in ('startseq', 'endseq')
        ]
        
        # Get ground truth references (remove startseq and endseq, split into list of words)
        img_refs = []
        for ref in captions_dict[img_name]:
            ref_clean = [
                w for w in ref.split() 
                if w not in ('startseq', 'endseq')
            ]
            img_refs.append(ref_clean)
            
        references.append(img_refs)
        hypotheses.append(generated)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total_val:
            print(f"Evaluated {idx + 1}/{total_val} images...")

    # 6. Calculate BLEU Scores
    print("\n📊 Calculating Corpus BLEU Scores...")
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("------------------------------------------")
    print(f"🔹 BLEU-1 Score: {bleu_1 * 100:.4f}")
    print(f"🔹 BLEU-2 Score: {bleu_2 * 100:.4f}")
    print(f"🔹 BLEU-3 Score: {bleu_3 * 100:.4f}")
    print(f"🔹 BLEU-4 Score: {bleu_4 * 100:.4f}")
    print("------------------------------------------")
    print("💡 Tip: A higher BLEU score indicates captions closer to the ground truth.")

if __name__ == '__main__':
    evaluate_model()
