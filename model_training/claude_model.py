import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy import sparse

# Load the dataset (assuming it's in a CSV file)
# Replace 'your_dataset.csv' with your actual file path
def load_data(file_path='figma_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
        df = df[~df['tag'].str.contains('-')]

    except:
        # If you're working with the sample data from the prompt
        columns = ['tag', 'type', 'width', 'height', 'characters', 'has_text', 
                   'depth', 'num_children', 'parent_tag', 'sibling_count', 
                   'is_leaf', 'font_size', 'has_font_size', 'font_name', 'has_text_color',
                   'color_r', 'color_g', 'color_b', 'has_background_color',
                   'background_r', 'background_g', 'background_b', 'border_radius',
                   'border_r', 'border_g', 'border_b', 'has_border', 'border_opacity',
                   'border_weight', 'has_shadow', 'shadow_r', 'shadow_g', 'shadow_b',
                   'shadow_radius', 'text_length', 'word_count', 'contains_number',
                   'contains_special_chars', 'x_quarter', 'y_quarter', 'aspect_ratio',
                   'area', 'normalized_width', 'normalized_height']
        
        data = [
            ['BODY', 'FRAME', 0.990183246, 1, 0, 0, 0.166666667, 0, 0, 0.333333333, 0, 'normal', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.172049124, 1, 1, 1],
            ['DIV', 'GROUP', 0.990183246, 0, 0, 1, 0, 'FRAME', 0.117647059, 1, 0.333333333, 0, 'normal', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0],
            ['DIV', 'GROUP', 0.990183246, 1, 0, 1, 0.055555556, 'FRAME', 0.117647059, 0, 0.333333333, 0, 'normal', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.172049124, 1, 1, 1],
            ['DIV', 'GROUP', 0.990183246, 1, 0, 2, 0.444444444, 'GROUP', 0, 0, 0.333333333, 0, 'normal', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.172049124, 1, 1, 1]
        ]
        
        df = pd.DataFrame(data, columns=columns)
    
    return df

# Preprocess the data
def preprocess_data(df):
    # Separate features and target
    X = df.drop('tag', axis=1)
    y = df['tag']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Prepare the target variable
    label_encoder = OneHotEncoder(sparse=False)
    y_encoded = label_encoder.fit_transform(y.values.reshape(-1, 1))
    
    # Split the data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Further split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert sparse matrices to dense if needed
    if sparse.issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()
    if sparse.issparse(X_val_processed):
        X_val_processed = X_val_processed.toarray()
    if sparse.issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor, label_encoder

# Build the neural network model
def build_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and evaluate the model
def train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # Get dimensions
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Build model
    model = build_model(input_dim, output_dim)
    
    # Define early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),  # Use explicit validation data instead of validation_split
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# Make predictions
def predict(model, preprocessor, label_encoder, features):
    # Preprocess the features
    processed_features = preprocessor.transform(pd.DataFrame([features]))
    
    # Convert to dense if needed
    if sparse.issparse(processed_features):
        processed_features = processed_features.toarray()
    
    # Make prediction
    prediction_proba = model.predict(processed_features)
    prediction_idx = np.argmax(prediction_proba, axis=1)
    
    # Decode the prediction
    prediction = label_encoder.inverse_transform(np.eye(len(label_encoder.categories_[0]))[prediction_idx])
    
    return prediction[0][0], prediction_proba[0][prediction_idx[0]]

# Main function to run the whole pipeline
def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Display data info
    print("\nDataset info:")
    print(f"Shape: {df.shape}")
    print(f"Number of unique tags: {df['tag'].nunique()}")
    print(f"Tags: {df['tag'].unique()}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, label_encoder = preprocess_data(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Train and evaluate model
    print("\nTraining model...")
    model, history = train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save model for future use
    model.save("html_tag_classifier.h5")
    print("\nModel saved as 'html_tag_classifier.h5'")

    return model, preprocessor, label_encoder

if __name__ == "__main__":
    main()