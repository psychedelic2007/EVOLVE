import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_score
from tqdm import tqdm
import logging
import argparse

class TransferLearningClassifier:
    def __init__(self, 
                 pretrained_model_path, 
                 dropout_rate=0.3, 
                 l2_lambda=0.001,
                 learning_rate=0.001):
        """
        Initialize Transfer Learning Classifier
        
        :param pretrained_model_path: Path to the pre-trained model
        :param dropout_rate: Dropout rate for regularization
        :param l2_lambda: L2 regularization strength
        :param learning_rate: Learning rate for optimizer
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate pretrained model path
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
        
        # Load pre-trained model
        self.pretrained_model = load_model(pretrained_model_path)
        self.input_shape = self.pretrained_model.input_shape[1:]
        self.logger.info(f"Inferred input shape: {self.input_shape}")
        
        # hyperparameters
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = self._create_transfer_model()
    
    def _create_transfer_model(self):
        """
        Create transfer learning model with dropout and L2 regularization
        
        :return: Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # Applying L2 regularization and dropout
        x = Dense(3, activation='tanh', kernel_regularizer=l2(self.l2_lambda))(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(8, activation='tanh', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(8, activation='tanh', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(self.dropout_rate)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Copy weights from pre-trained model
        model.set_weights(self.pretrained_model.get_weights())
        
        # Freeze most layers
        for layer in model.layers[:-2]:
            layer.trainable = False
        
        # Compile grad clipping
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def load_and_preprocess_data(self, data_dir):
        """
        Load and preprocess data from CSV files
        
        :param data_dir: Directory containing training CSV files
        :return: Preprocessed features and labels
        """
        self.logger.info(f"Loading data from {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Invalid directory: {data_dir}")
        
        file_paths = [
            os.path.join(data_dir, file) 
            for file in os.listdir(data_dir) 
            if file.endswith('.csv')
        ]
        
        if not file_paths:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        data_list = []
        for file_path in tqdm(file_paths, desc="Loading Files"):
            data = pd.read_csv(file_path)
            data_list.append(data)
        
        combined_data = pd.concat(data_list, ignore_index=True)
        
        # Preprocessing
        X = combined_data.drop('Target', axis=1)
        y = combined_data['Target']
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        return X_scaled, y.values.reshape(-1, 1)
    
    def perform_loocv(self, X, y):
        """
        Perform Leave-One-Out Cross-Validation
        
        :param X: Input features
        :param y: Target labels
        :return: Cross-validation scores
        """
        self.logger.info("Performing Leave-One-Out Cross-Validation")
        
        loo = LeaveOneOut()
        cv_scores = cross_val_score(
            self.model, X, y, 
            scoring='accuracy', 
            cv=loo
        )
        
        self.logger.info(f"LOOCV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return cv_scores
    
    def train_model(self, X, y, X_val=None, y_val=None):
        """
        Train the transfer learning model
        
        :param X: Training features
        :param y: Training labels
        :param X_val: Validation features
        :param y_val: Validation labels
        :return: Training history
        """
        self.logger.info("Starting model training")
        
        # Use validation data if provided, otherwise split
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        # Training with progress bar
        history = self.model.fit(
            X_train, y_train,
            epochs=500,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0,  # Silent mode
            callbacks=[
                tf.keras.callbacks.ProgbarLogger(count_mode='samples')
            ]
        )
        return history
    
    def save_model(self, save_path):
        """
        Save the trained model
        
        :param save_path: Path to save the model
        """
        self.model.save(save_path, save_format='h5')
        self.logger.info(f"Model saved to {save_path}")
    
    def predict_mutation(self, sequence):
        """
        Predict mutation probability
        
        :param sequence: Input sequence
        :return: Mutation probability
        """
        scaler = StandardScaler()
        sequence_scaled = scaler.fit_transform([sequence])
        prediction = self.model.predict(sequence_scaled)
        return prediction[0][0]

def get_user_inputs():
    """
    Interactively get user inputs for model paths
    
    :return: Tuple of (pretrained_model_path, new_data_dir, save_model_path)
    """
    print("\n--- Transfer Learning Model Configuration ---")
    
    # Pretrained Model Path
    while True:
        pretrained_model_path = input("Enter the path to the pre-trained model: ").strip()
        if os.path.exists(pretrained_model_path):
            break
        print("Error: File not found. Please check the path.")
    
    while True:
        new_data_dir = input("Enter the directory path containing training files (CSV): ").strip()
        if os.path.isdir(new_data_dir):
            csv_files = [f for f in os.listdir(new_data_dir) if f.endswith('.csv')]
            if csv_files:
                break
            print("Error: No CSV files found in the directory.")
        else:
            print("Error: Invalid directory path.")
    
    # Save Model Path
    while True:
        save_model_path = input("Enter the path to save the transfer-learned model (e.g., model.h5): ").strip()
        try:
            save_dir = os.path.dirname(save_model_path) or '.'
            if os.access(save_dir, os.W_OK):
                break
            print("Error: Cannot write to the specified directory.")
        except Exception as e:
            print(f"Error: {e}")
    
    return pretrained_model_path, new_data_dir, save_model_path

def main():
    PRETRAINED_MODEL_PATH, NEW_DATA_DIR, SAVE_MODEL_PATH = get_user_inputs()
    
    try:
        # Initialize Transfer Learning Classifier
        tl_classifier = TransferLearningClassifier(
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            dropout_rate=0.3,
            l2_lambda=0.001,
            learning_rate=0.001
        )
        
        X, y = tl_classifier.load_and_preprocess_data(NEW_DATA_DIR)
        cv_scores = tl_classifier.perform_loocv(X, y)
        tl_classifier.save_model(SAVE_MODEL_PATH)
        
        print("\n--- Training Complete ---")
        print(f"Model saved to: {SAVE_MODEL_PATH}")
        print(f"LOOCV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
