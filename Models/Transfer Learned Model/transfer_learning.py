import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, Reshape, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from tqdm import tqdm
import logging
import argparse


class AdaptivePooling(Layer):
    """
    Custom layer for adaptive pooling to handle variable sequence lengths
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AdaptivePooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(AdaptivePooling, self).build(input_shape)
        
    def call(self, x):
        # Reshape if necessary based on input dimensions
        input_shape = tf.shape(x)
        
        # Handle different input shapes by first reshaping to a sequence
        if len(x.shape) <= 2:  # For 1D or 2D inputs
            # Reshape to (batch_size, seq_length, features)
            reshaped = Reshape((input_shape[1], 1))(x)
        else:
            reshaped = x
            
        # Apply global average pooling
        pooled = tf.reduce_mean(reshaped, axis=1)
        
        return pooled
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class TransferLearningClassifier:
    def __init__(self, 
                 pretrained_model_path, 
                 dropout_rate=0.3, 
                 l2_lambda=0.001,
                 learning_rate=0.001,
                 use_semi_supervised=True,
                 unlabeled_weight=0.5):
        """
        Initialize Transfer Learning Classifier with advanced features
        
        :param pretrained_model_path: Path to the pre-trained model
        :param dropout_rate: Dropout rate for regularization
        :param l2_lambda: L2 regularization strength
        :param learning_rate: Learning rate for optimizer
        :param use_semi_supervised: Whether to use semi-supervised learning
        :param unlabeled_weight: Weight for unlabeled data in semi-supervised learning
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate pretrained model path
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
        
        # Load pre-trained model
        self.pretrained_model = load_model(pretrained_model_path, 
                                          custom_objects={'AdaptivePooling': AdaptivePooling})
        self.input_shape = self.pretrained_model.input_shape[1:]
        self.logger.info(f"Inferred input shape: {self.input_shape}")
        
        # Hyperparameters
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.use_semi_supervised = use_semi_supervised
        self.unlabeled_weight = unlabeled_weight
        
        # Preprocessing objects
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = self._create_transfer_model()
        
        # For semi-supervised learning
        if use_semi_supervised:
            self.autoencoder = self._create_autoencoder()
    
    def _create_transfer_model(self):
        """
        Create transfer learning model with adaptive pooling, dropout and L2 regularization
        
        :return: Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # Add adaptive pooling to handle variable sequence lengths
        # This transforms inputs of any length to a fixed representation
        x = AdaptivePooling(output_dim=self.input_shape[-1])(inputs)
        
        # Applying L2 regularization and dropout
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(8, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(8, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(self.dropout_rate)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Copy weights from pre-trained model for compatible layers
        # We need to be careful here due to the architectural differences
        pretrained_weights = self.pretrained_model.get_weights()
        model_weights = model.get_weights()
        
        # Only copy weights for layers with matching shapes
        for i, (p_layer, m_layer) in enumerate(zip(self.pretrained_model.layers, model.layers)):
            if p_layer.name == m_layer.name and p_layer.count_params() == m_layer.count_params():
                model_weights[i] = pretrained_weights[i]
                
        model.set_weights(model_weights)
        
        # Freeze most layers
        for layer in model.layers[:-3]:  # Keep last 3 layers trainable
            layer.trainable = False
        
        # Compile with gradient clipping
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_autoencoder(self):
        """
        Create autoencoder for semi-supervised pre-training
        
        :return: Compiled autoencoder model
        """
        # Encoder
        inputs = Input(shape=self.input_shape)
        x = AdaptivePooling(output_dim=self.input_shape[-1])(inputs)
        
        # Bottleneck
        encoded = Dense(8, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        
        # Decoder
        decoded = Dense(self.input_shape[-1], activation='linear')(encoded)
        
        # Full autoencoder
        autoencoder = Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def load_and_preprocess_data(self, data_dir, unlabeled_dir=None):
        """
        Load and preprocess data from CSV files, including unlabeled data
        
        :param data_dir: Directory containing labeled training CSV files
        :param unlabeled_dir: Directory containing unlabeled data files (optional)
        :return: Preprocessed features and labels
        """
        self.logger.info(f"Loading labeled data from {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Invalid directory: {data_dir}")
        
        # Load labeled data
        labeled_paths = [
            os.path.join(data_dir, file) 
            for file in os.listdir(data_dir) 
            if file.endswith('.csv')
        ]
        
        if not labeled_paths:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        labeled_data_list = []
        for file_path in tqdm(labeled_paths, desc="Loading Labeled Files"):
            data = pd.read_csv(file_path)
            labeled_data_list.append(data)
        
        labeled_data = pd.concat(labeled_data_list, ignore_index=True)
        
        # Preprocessing labeled data
        X = labeled_data.drop('Target', axis=1)
        y = labeled_data['Target']
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Load unlabeled data if provided
        if unlabeled_dir and os.path.isdir(unlabeled_dir) and self.use_semi_supervised:
            self.logger.info(f"Loading unlabeled data from {unlabeled_dir}")
            unlabeled_paths = [
                os.path.join(unlabeled_dir, file) 
                for file in os.listdir(unlabeled_dir) 
                if file.endswith('.csv')
            ]
            
            if unlabeled_paths:
                unlabeled_data_list = []
                for file_path in tqdm(unlabeled_paths, desc="Loading Unlabeled Files"):
                    data = pd.read_csv(file_path)
                    # Remove Target column if exists
                    if 'Target' in data.columns:
                        data = data.drop('Target', axis=1)
                    unlabeled_data_list.append(data)
                
                if unlabeled_data_list:
                    unlabeled_data = pd.concat(unlabeled_data_list, ignore_index=True)
                    X_unlabeled_imputed = self.imputer.transform(unlabeled_data)
                    X_unlabeled_scaled = self.scaler.transform(X_unlabeled_imputed)
                    
                    return X_scaled, y.values.reshape(-1, 1), X_unlabeled_scaled
        
        return X_scaled, y.values.reshape(-1, 1), None
    
    def perform_loocv(self, X, y):
        """
        Perform Leave-One-Out Cross-Validation
        
        :param X: Input features
        :param y: Target labels
        :return: Cross-validation scores
        """
        self.logger.info("Performing Leave-One-Out Cross-Validation")
        
        loo = LeaveOneOut()
        cv_scores = []
        
        for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="LOOCV Progress"):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Reset model for each fold
            self.model = self._create_transfer_model()
            
            # Train model
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Evaluate
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            cv_scores.append(accuracy)
        
        cv_scores = np.array(cv_scores)
        self.logger.info(f"LOOCV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return cv_scores
    
    def pretrain_with_unlabeled_data(self, X_unlabeled, epochs=50):
        """
        Pretrain the model with unlabeled data using autoencoder
        
        :param X_unlabeled: Unlabeled data features
        :param epochs: Number of pretraining epochs
        """
        if not self.use_semi_supervised or X_unlabeled is None:
            return
        
        self.logger.info("Pretraining with unlabeled data")
        
        # Train autoencoder on unlabeled data
        self.autoencoder.fit(
            X_unlabeled, X_unlabeled,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            verbose=1
        )
        
        # Extract encoder weights
        encoder_layers = [layer for layer in self.autoencoder.layers if 'dense' in layer.name][:1]
        
        # Transfer weights to main model
        for i, layer in enumerate(encoder_layers):
            # Find corresponding layer in main model
            for main_layer in self.model.layers:
                if main_layer.name.startswith('dense') and main_layer.units == layer.units:
                    self.logger.info(f"Transferring weights from autoencoder layer {layer.name} to model layer {main_layer.name}")
                    main_layer.set_weights(layer.get_weights())
                    break
    
    def train_model(self, X, y, X_unlabeled=None, validation_split=0.15, epochs=100):
        """
        Train the transfer learning model with optional semi-supervised pretraining
        
        :param X: Training features
        :param y: Training labels
        :param X_unlabeled: Unlabeled features for semi-supervised learning
        :param validation_split: Percentage of data to use for validation
        :param epochs: Number of training epochs
        :return: Training history
        """
        self.logger.info("Starting model training")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # If we have unlabeled data and semi-supervised is enabled, pretrain
        if self.use_semi_supervised and X_unlabeled is not None:
            self.pretrain_with_unlabeled_data(X_unlabeled)
        
        # Fine-tune on labeled data
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Final evaluation
        loss, accuracy = self.model.evaluate(X_val, y_val)
        self.logger.info(f"Final validation accuracy: {accuracy:.4f}")
        
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
        # Reshape for adaptive pooling if needed
        sequence_processed = self.scaler.transform([sequence])
        prediction = self.model.predict(sequence_processed)
        return prediction[0][0]


def get_user_inputs():
    """
    Interactively get user inputs for model paths
    
    :return: Tuple of (pretrained_model_path, labeled_data_dir, unlabeled_data_dir, save_model_path)
    """
    print("\n--- Advanced Transfer Learning Model Configuration ---")
    
    # Pretrained Model Path
    while True:
        pretrained_model_path = input("Enter the path to the pre-trained model: ").strip()
        if os.path.exists(pretrained_model_path):
            break
        print("Error: File not found. Please check the path.")
    
    # Labeled Data Directory
    while True:
        labeled_data_dir = input("Enter the directory path containing labeled training files (CSV): ").strip()
        if os.path.isdir(labeled_data_dir):
            csv_files = [f for f in os.listdir(labeled_data_dir) if f.endswith('.csv')]
            if csv_files:
                break
            print("Error: No CSV files found in the directory.")
        else:
            print("Error: Invalid directory path.")
    
    # Unlabeled Data Directory (Optional)
    use_unlabeled = input("Do you want to use unlabeled data for semi-supervised learning? (y/n): ").strip().lower()
    unlabeled_data_dir = None
    
    if use_unlabeled == 'y':
        while True:
            unlabeled_data_dir = input("Enter the directory path containing unlabeled files (CSV): ").strip()
            if not unlabeled_data_dir:
                print("Skipping unlabeled data.")
                break
                
            if os.path.isdir(unlabeled_data_dir):
                csv_files = [f for f in os.listdir(unlabeled_data_dir) if f.endswith('.csv')]
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
    
    return pretrained_model_path, labeled_data_dir, unlabeled_data_dir, save_model_path


def main():
    # Get user inputs
    PRETRAINED_MODEL_PATH, LABELED_DATA_DIR, UNLABELED_DATA_DIR, SAVE_MODEL_PATH = get_user_inputs()
    
    try:
        # Initialize Advanced Transfer Learning Classifier
        tl_classifier = TransferLearningClassifier(
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            dropout_rate=0.3,
            l2_lambda=0.001,
            learning_rate=0.001,
            use_semi_supervised=(UNLABELED_DATA_DIR is not None)
        )
        
        # Load and preprocess data
        X, y, X_unlabeled = tl_classifier.load_and_preprocess_data(
            LABELED_DATA_DIR, 
            unlabeled_dir=UNLABELED_DATA_DIR
        )
        
        # Train model
        if len(X) > 50:  # For larger datasets, use train/test split
            tl_classifier.train_model(X, y, X_unlabeled=X_unlabeled)
        else:  # For smaller datasets, use LOOCV
            cv_scores = tl_classifier.perform_loocv(X, y)
            print(f"LOOCV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save the trained model
        tl_classifier.save_model(SAVE_MODEL_PATH)
        
        print("\n--- Training Complete ---")
        print(f"Model saved to: {SAVE_MODEL_PATH}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
