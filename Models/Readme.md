# Models Repository

## Overview
This repository contains machine learning models categorized into two folders: **Pre-trained Models** and **Transfer Learned Models**. These models are used for various sequence-based predictions and can be utilized directly or fine-tuned for specific tasks.

## Directory Structure
```
📂 Models
   ├── 📂 Pre-trained model
   │   ├── ctv_model.h5
   │   ├── influenza_model.h5
   │   ├── omicron_model.h5
   │   ├── xanthomonas_model.h5
   │   ├── model.py
   ├── 📂 Transfer learned model
   │   ├── dengue_model.h5
   │   ├── ebola_fine_tuned_model.h5
   │   ├── predictions_using_transfer_learned_model.py
   │   ├── transfer_learning.py
```

## Pre-trained Models
These models have been trained on large datasets and can be used for inference or further fine-tuning.

### 1. **CTV Model** (`ctv_model.h5`)
   - Pre-trained model for CTV sequence predictions.

### 2. **Influenza Model** (`influenza_model.h5`)
   - Model trained for predicting influenza-related sequence outcomes.

### 3. **Omicron Model** (`omicron_model.h5`)
   - Specialized model for Omicron variant analysis.

### 4. **Xanthomonas Model** (`xanthomonas_model.h5`)
   - Model trained for analyzing Xanthomonas sequences.

### 5. **Model Training Script** (`model.py`)
   - Python script demonstrating how the pre-trained models were originally trained.

## Transfer Learned Models
These models have been fine-tuned using transfer learning techniques to adapt to specific datasets.

### 1. **Dengue Model** (`dengue_model.h5`)
   - Transfer learned model for Dengue virus sequence analysis.

### 2. **Ebola Fine-Tuned Model** (`ebola_fine_tuned_model.h5`)
   - Fine-tuned model for Ebola sequence classification.

### 3. **Prediction Script for Transfer Learned Models** (`predictions_using_transfer_learned_model.py`)
   - Python script to make predictions using transfer-learned models.

### 4. **Transfer Learning Training Script** (`transfer_learning.py`)
   - Script demonstrating how the transfer learning process was performed.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/models.git
   cd models
   ```
2. Load a model and make predictions:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('Pre-trained model/ctv_model.h5')
   ```
3. Use the transfer learning script to fine-tune models:
   ```bash
   python Transfer\ learned\ model/transfer_learning.py
   ```
4. Make predictions using the transfer-learned models:
   ```bash
   python Transfer\ learned\ model/predictions_using_transfer_learned_model.py
   ```

## Contributions
Contributions are welcome! If you have additional models or improvements, feel free to submit a pull request.

## License
This repository is available under [MIT License](LICENSE).

## Contact
For any inquiries, please reach out via [webserver.iiserkol@gmail.com] or create an issue in this repository.
