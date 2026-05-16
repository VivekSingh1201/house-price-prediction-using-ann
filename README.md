# House Price Prediction using Artificial Neural Network (ANN)

## Overview
This project implements a **House Price Prediction System using Artificial Neural Networks (ANN)** with TensorFlow/Keras. The goal is to predict house sale prices based on structured tabular housing data containing both numerical and categorical features.

This project demonstrates the complete deep learning workflow:
- Data preprocessing
- Missing value handling
- Categorical feature encoding
- Feature normalization
- ANN model building
- Hyperparameter tuning
- Regularization techniques
- Dropout
- Early stopping
- Learning rate experimentation
- TensorBoard visualization
- Model evaluation
- Model persistence

## Problem Statement
Predict house sale prices from housing features using supervised regression.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorBoard
- Jupyter Notebook

## Project Structure
```text
House-Price-Prediction-ANN/
│
├── train.csv
├── main.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
```bash
git clone https://github.com/your-username/House-Price-Prediction-ANN.git
cd House-Price-Prediction-ANN
python3 -m venv DLenv
source DLenv/bin/activate
pip install -r requirements.txt
```

## Dataset
- 1460 rows
- 80+ columns
- 43 categorical columns
- Target: `SalePrice`

## Data Preprocessing

### Feature and Target Split
```python
trainX = train.iloc[:, :-1]
trainY = train.iloc[:, -1]
```

### Identify Categorical Columns
```python
categorical_cols = trainX.select_dtypes(include=['object', 'string']).columns
```

### Missing Value Handling
Categorical columns:
```python
trainX[categorical_cols] = trainX[categorical_cols].fillna(
    trainX[categorical_cols].mode().iloc[0]
)
```

Numerical columns:
```python
trainX = trainX.fillna(trainX.median())
```

Why imputation instead of dropping rows?
- preserves dataset size
- avoids information loss
- improves ANN learning

## One-Hot Encoding
```python
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = ohe.fit_transform(trainX[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_features,
    columns=ohe.get_feature_names_out(categorical_cols)
)
```

## Merge Encoded Features
```python
trainX = trainX.drop(columns=categorical_cols)

trainX = pd.concat(
    [
        trainX.reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)
```

## Feature Scaling
```python
feature_scaler = MinMaxScaler()
norm_trainX = feature_scaler.fit_transform(trainX)
```

## Target Scaling
```python
target_scaler = MinMaxScaler()
norm_trainY = target_scaler.fit_transform(trainY.values.reshape(-1,1))
```

## Train-Test Split
```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42
)
```

## ANN Architecture
```text
Input Features
      │
      ▼
Dense (256, ReLU)
      ▼
Dense (128, ReLU)
      ▼
Dense (64, ReLU)
      ▼
Dense (32, ReLU)
      ▼
Dense (16, ReLU)
      ▼
Output (1, Linear)
```

## Model Creation
```python
model = Sequential([
    Input(shape=(norm_trainX.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
```

## Compilation
```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

## Hyperparameter Experiments
Tested:
- Optimizers: Adam, SGD, RMSprop, Adagrad, Adadelta
- Activations: linear, sigmoid, tanh, relu, softmax
- Bias initialization
- L1/L2/Elastic Net regularization
- Dropout
- Early stopping
- Cyclic learning rate

Best observed:
- Activation: ReLU
- Optimizer: Adam

## Training
```python
history = model.fit(
    norm_trainX,
    norm_trainY,
    batch_size=128,
    epochs=50,
    validation_split=0.2,
    verbose=1
)
```

## Testing Pipeline
```python
predictions = model.predict(norm_test)
final_predictions = target_scaler.inverse_transform(predictions)
```

Evaluation metrics:
- MAE
- MSE
- RMSE

## TensorBoard
```bash
tensorboard --logdir=.
```

Open:
```text
http://localhost:6006
```

## Model Saving
```python
model.save('optimum_model.keras')
```

## Model Loading
```python
from tensorflow.keras.models import load_model
model = load_model('optimum_model.keras', compile=False)
```

## Common Errors Fixed
- `predict_classes` removed → use `model.predict()`
- feature mismatch between train/test
- 3D prediction tensor reshape issues
- optimizer loading warnings
- TensorBoard environment dependency issues

## Future Improvements
- XGBoost comparison
- Batch normalization
- Keras Tuner
- Cross-validation
- Flask/Streamlit deployment

## Requirements
```txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
tensorboard
jupyter
ipykernel
```

## Author
**Vivek Kumar Singh**  
