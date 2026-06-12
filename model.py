import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
import config

# =====================================================================
# 1. OPTIMIZED MERGE MODEL (For Global Features of shape (2048,))
# =====================================================================
def build_merge_model(vocab_size, max_length):
    """
    Builds the optimized merge-architecture image captioning model.
    Compatible with global feature vectors of shape (2048,).
    Uses Keras Functional API, supports masking, and is compiled with sparse crossentropy.
    """
    # Image feature input
    img_input = Input(shape=(config.IMAGE_FEATURE_DIM,), name="image_input")
    img_feat = Dropout(0.5, name="image_dropout")(img_input)
    img_feat = Dense(config.DENSE_UNITS, activation='relu', name="image_dense")(img_feat)
    
    # Text sequence input
    txt_input = Input(shape=(max_length,), name="text_input")
    # mask_zero=True tells downstream layers (LSTM) to ignore padded zeros
    txt_feat = Embedding(
        input_dim=vocab_size, 
        output_dim=config.EMBEDDING_DIM, 
        mask_zero=True, 
        name="text_embedding"
    )(txt_input)
    txt_feat = Dropout(0.5, name="text_dropout")(txt_feat)
    txt_feat = LSTM(config.DECODER_UNITS, name="text_lstm")(txt_feat)
    
    # Merge branch (decoder)
    decoder = add([img_feat, txt_feat], name="merge_layer")
    decoder = Dense(config.DENSE_UNITS, activation='relu', name="decoder_dense")(decoder)
    output = Dense(vocab_size, activation='softmax', name="decoder_output")(decoder)
    
    model = Model(inputs=[img_input, txt_input], outputs=output, name="Optimized_Merge_Captioner")
    
    return model


# =====================================================================
# 2. BAHDANAU ATTENTION-BASED ENCODER-DECODER (For Spatial Features)
# =====================================================================
class CNN_Encoder(tf.keras.Model):
    """
    Projects CNN spatial feature maps (e.g. shape (batch_size, 64, 2048))
    into a lower-dimensional embedding space.
    """
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, x):
        # Shape change: (batch_size, 64, 2048) -> (batch_size, 64, embedding_dim)
        x = self.fc(x)
        return x


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau (Additive) Attention Layer.
    Calculates weights for each spatial feature position based on the LSTM hidden state.
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch_size, 64, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        # Expand hidden state to add time axis: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape: (batch_size, 64, 1)
        # Calculates alignment scores between hidden state and image features
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # attention_weights shape: (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape: (batch_size, embedding_dim)
        # Multiply features by weights and sum over spatial dimension
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    """
    Decoder model that uses Bahdanau Attention to sequentially generate caption words.
    """
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc1 = tf.keras.layers.Dense(units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden):
        # x shape: (batch_size, 1) (single word index input)
        # features shape: (batch_size, 64, embedding_dim)
        # hidden shape: (batch_size, units)
        
        # Calculate attention context vector
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding: (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Concatenate embedded word and context vector along features axis
        # context_vector expanded: (batch_size, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass combined vector to LSTM
        output, state, _ = self.lstm(x, initial_state=[hidden, hidden])

        # Output shape: (batch_size, 1, units)
        x = self.fc1(output)

        # Reshape output to (batch_size, units)
        x = tf.reshape(x, (-1, x.shape[2]))

        # Output shape: (batch_size, vocab_size)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
