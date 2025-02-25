import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset with appropriate encoding to avoid decoding errors
df_all = pd.read_csv('figma_dataset.csv')

# Remove unwanted tags
df = df_all[~df_all['tag'].str.contains('-', na=False)]
df = df[~df['tag'].str.contains('BODY', na=False)]

# Save cleaned dataset (optional)
df.to_csv('cleaned_figma_dataset.csv', index=False)

# Define column categories based on the dataset attributes
categorical_cols = ["type", "parent_tag", "characters", "font_name"]
numerical_cols = ['width', 'height', 'has_text', 'depth', 'num_children', 'sibling_count', 'is_leaf', 
                  'font_size', 'has_font_size', 'has_text_color', 'color_r', 'color_g', 'color_b', 
                  'has_background_color', 'background_r', 'background_g', 'background_b', 'border_radius', 
                  'border_r', 'border_g', 'border_b', 'border_weight', 'has_shadow', 'shadow_r', 'shadow_g', 
                  'shadow_b', 'shadow_radius', 'text_length', 'word_count', 'contains_number', 
                  'contains_special_chars', 'has_border', 'border_opacity', 'x_quarter', 'y_quarter', 
                  'aspect_ratio', 'area', 'normalized_width', 'normalized_height']

# Convert numerical columns to numeric and handle conversion errors
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values with median for numerical columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Ensure all categorical columns are treated as strings before encoding
df[categorical_cols] = df[categorical_cols].astype(str)

# Encode categorical features using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
tag_encoder = LabelEncoder()
df['tag'] = tag_encoder.fit_transform(df['tag'])

# Check class distribution before applying SMOTE
print("Class Distribution Before SMOTE:\n", df['tag'].value_counts())

# Remove extremely rare classes (less than 2 samples)
class_counts = df['tag'].value_counts()
rare_classes = class_counts[class_counts < 2].index
df = df[~df['tag'].isin(rare_classes)]

# Split dataset into features and target
X = df.drop(columns=['tag'])  # Features
y = df['tag']  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Apply SMOTE with adjusted k_neighbors
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if a pre-trained model exists
model_filename = "models/html_tag_model.pkl"
scaler_filename = "models/scaler.pkl"
tag_encoder_filename = "models/tag_encoder.pkl"
label_encoders_filename = "models/label_encoders.pkl"

try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    tag_encoder = joblib.load(tag_encoder_filename)
    label_encoders = joblib.load(label_encoders_filename)
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("No pre-trained model found, training a new one.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model, scaler, and encoders
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(tag_encoder, tag_encoder_filename)
    joblib.dump(label_encoders, label_encoders_filename)
    print("Model saved for future use.")

# Predict and evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=tag_encoder.inverse_transform(np.unique(y_test))))
