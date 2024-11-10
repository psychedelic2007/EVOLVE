import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Custom loss function to avoid log(0)
def custom_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Stating a Function to recreate the model architecture
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(3, activation='tanh')(inputs)
    x = Dense(8, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# pretrained model
pretrained_model = load_model('<pre-trained model name>')
input_shape = pretrained_model.input_shape[1:]
print(f"Inferred input shape: {input_shape}")

# Creating new model - keeping the same architecture
base_model = create_model(input_shape)

# Copying weights from pretrained model (Important step!!)
base_model.set_weights(pretrained_model.get_weights())

# Freezing all layers except the last one
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Compiling the model with gradient clipping
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
base_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# File loading
def load_and_preprocess_data(file_paths):
    data_list = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        data_list.append(data)

    combined_data = pd.concat(data_list, ignore_index=True)

    X = combined_data.drop('Target', axis=1)
    y = combined_data['Target']

    # Handling NaN values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y.values.reshape(-1, 1)

new_data_dir = 'training_files'
file_paths = [os.path.join(new_data_dir, file) for file in os.listdir(new_data_dir) if file.endswith('.csv')]

X, y = load_and_preprocess_data(file_paths)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
history = base_model.fit(X_train, y_train,
                        epochs=500,
                        batch_size=32,
                        validation_data=(X_val, y_val))

# Saving model
base_model.save('<name_of_transfer_learned_model.h5>', save_format='h5')

# Predictions
def predict_mutation(sequence):
    scaler = StandardScaler()
    sequence_scaled = scaler.fit_transform([sequence])
    prediction = base_model.predict(sequence_scaled)
    return prediction[0][0]
