import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import argparse

class AdvancedModelTrainer:
    def __init__(self, 
                 learning_rate=1e-4, 
                 dropout_rate=0.3, 
                 random_state=42):
        """
        Initialize the Advanced Model Trainer
        
        :param learning_rate: Initial learning rate
        :param dropout_rate: Dropout rate for regularization
        :param random_state: Random seed for reproducibility
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # random seed
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Model and scaler will be set during training
        self.model = None
        self.scaler = None
    
    def custom_binary_crossentropy(self, y_true, y_pred):
        """
        Custom binary cross-entropy loss function with numerical stability
        
        :param y_true: True labels
        :param y_pred: Predicted probabilities
        :return: Loss value
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred + epsilon) +(1 - y_true) * tf.math.log(1 - y_pred + epsilon))
    
    def create_model(self, input_shape):
        """
        Create neural network model with regularization
        
        :param input_shape: Shape of input features
        :return: Compiled Keras model
        """
        model = Sequential([
            Dense(3, activation='tanh', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(self.dropout_rate),
            Dense(8, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(self.dropout_rate),
            Dense(8, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss=self.custom_binary_crossentropy, 
            metrics=['accuracy']
        )
        
        return model
    
    def load_and_preprocess_data(self, data_dir, feature_columns):
        """
        Load and preprocess data from multiple CSV files
        
        :param data_dir: Directory containing CSV files
        :param feature_columns: List of feature column names
        :return: Preprocessed features and labels
        """
        self.logger.info(f"Loading data from {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Invalid directory: {data_dir}")
        
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        # Load and combine data
        data_list = []
        for filename in tqdm(all_files, desc="Loading Files"):
            df = pd.read_csv(filename)
            
            required_columns = feature_columns + ['Target']
            if not all(col in df.columns for col in required_columns):
                self.logger.warning(f"Skipping {filename}: Missing required columns")
                continue
            
            # Clean data
            df = df.dropna(subset=required_columns)
            data_list.append(df)
        
        # Combine datasets
        combined_df = pd.concat(data_list, ignore_index=True)
        self.logger.info(f"Combined data shape: {combined_df.shape}")
        
        # Prepare features and labels
        X = combined_df[feature_columns].values
        y = combined_df['Target'].values
        
        # Handle potential NaN or infinite values
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.3):
        """
        Train the model with advanced techniques
        
        :param X: Input features
        :param y: Target labels
        :param test_size: Proportion of validation data
        :return: Training history
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.random_state)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Clip extreme values
        X_train = np.clip(X_train, -5, 5)
        X_val = np.clip(X_val, -5, 5)
        
        # Create model
        self.model = self.create_model(input_shape=(X_train.shape[1],))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return history
    
    def cross_validate(self, X, y, n_splits=5):
      """
      Perform cross-validation manually
    
      :param X: Input features
      :param y: Target labels
      :param n_splits: Number of cross-validation splits
      :return: Cross-validation scores
      """
      # Prepare cross-validation
      kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
    
      cv_scores = []
      for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        self.logger.info(f"Training fold {fold+1}/{n_splits}")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        # Clip extreme values
        X_train_fold = np.clip(X_train_fold, -5, 5)
        X_val_fold = np.clip(X_val_fold, -5, 5)
        
        # Create and train a new model for this fold
        fold_model = self.create_model(input_shape=(X_train_fold.shape[1],))
        
        # Early stopping and LR reduction for each fold
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,  # Reduced for CV
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3,  # Reduced for CV
            min_lr=1e-6
        )
        
        # Train
        fold_model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,  # Reduced for CV
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        _, accuracy = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(accuracy)
        self.logger.info(f"Fold {fold+1}/{n_splits} - Accuracy: {accuracy:.4f}")
    
      cv_scores = np.array(cv_scores)
      self.logger.info(f"Cross-validation scores: {cv_scores}")
      self.logger.info(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
      return cv_scores
    
    def save_model(self, save_path):
        """
        Save trained model
        
        :param save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(save_path)
        self.logger.info(f"Model saved to {save_path}")

def get_user_inputs():
    """
    Interactively get user inputs for model training
    
    :return: Dictionary of configuration parameters
    """
    print("\n--- Model Training Configuration ---")
    
    while True:
        data_dir = input("Enter the directory path containing training files (CSV): ").strip()
        if os.path.isdir(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                break
            print("Error: No CSV files found in the directory.")
        else:
            print("Error: Invalid directory path.")
    
    # Feature Columns
    while True:
        feature_input = input("Enter feature column names (comma-separated, e.g., F1,F3,F4): ").strip()
        feature_columns = [col.strip() for col in feature_input.split(',')]
        
        sample_file = os.path.join(data_dir, csv_files[0])
        sample_df = pd.read_csv(sample_file)
        
        if all(col in sample_df.columns for col in feature_columns):
            break
        print("Error: One or more feature columns not found in the CSV files.")
    
    # Save Model Path
    while True:
        save_model_path = input("Enter the path to save the trained model (e.g., model.h5): ").strip()
        try:
            save_dir = os.path.dirname(save_model_path) or '.'
            if os.access(save_dir, os.W_OK):
                break
            print("Error: Cannot write to the specified directory.")
        except Exception as e:
            print(f"Error: {e}")
    
    return {
        'data_dir': data_dir,
        'feature_columns': feature_columns,
        'save_model_path': save_model_path
    }

def main():
    config = get_user_inputs()
    
    try:
        trainer = AdvancedModelTrainer(
            learning_rate=1e-4,
            dropout_rate=0.3,
            random_state=42
        )
        
        # Load and preprocess data
        X, y = trainer.load_and_preprocess_data(
            config['data_dir'], 
            config['feature_columns']
        )
        
        history = trainer.train_model(X, y)
        cv_scores = trainer.cross_validate(X, y)
        trainer.save_model(config['save_model_path'])
        
        print("\n--- Training Complete ---")
        print(f"Model saved to: {config['save_model_path']}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
