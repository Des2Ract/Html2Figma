import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
# Load dataset
df_all = pd.read_csv('figma_dataset.csv')

df = df_all[~df_all['tag'].str.contains('-', na=False)]
# Separate features and target (assume 'tag' is the target)
X = df.drop(columns=["tag"])
y = df["tag"]

# Define categorical and numerical columns.
categorical_cols = ["type", "parent_tag", "characters", "font_name"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Fill missing values: for numerical columns, fill with 0; for categorical, use a placeholder.
X[numerical_cols] = X[numerical_cols].fillna(0)
X[categorical_cols] = X[categorical_cols].fillna("missing")

# Scale numerical features.
scaler_num = StandardScaler()
X_num_scaled = scaler_num.fit_transform(X[numerical_cols])

# Process categorical features with LabelEncoder.
# We encode each categorical column separately and then scale these encoded values.
cat_encoders = {}
X_cat = X[categorical_cols].copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col].astype(str))
    cat_encoders[col] = le

# Scale the label-encoded categorical features.
scaler_cat = StandardScaler()
X_cat_scaled = scaler_cat.fit_transform(X_cat.values)

# Combine numerical and categorical features.
X_processed = np.concatenate([X_num_scaled, X_cat_scaled], axis=1)

# Encode target labels with LabelEncoder.
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Save the preprocessors and encoders for future use.
joblib.dump(scaler_num, "scaler_num.pkl")
joblib.dump(scaler_cat, "scaler_cat.pkl")
joblib.dump(cat_encoders, "cat_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

# ---------------------------
# 2. Train/Test Split & Tensor Conversion
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Compute class weights to help with imbalanced classes.
classes = np.unique(y_encoded)
class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ---------------------------
# 3. Define the Neural Network Model
# ---------------------------
class TagClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(TagClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

input_size = X_train_tensor.shape[1]
output_size = len(target_encoder.classes_)
model = TagClassifier(input_size, output_size)

# ---------------------------
# 4. Training Setup
# ---------------------------
# Define loss function with class weights and the optimizer.
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5. Training Loop
# ---------------------------
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model.
torch.save(model.state_dict(), "tag_classifier.pth")

# ---------------------------
# 6. Evaluation
# ---------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, dim=1).numpy()

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\nAccuracy: {accuracy:.4f}")
unique_classes = np.unique(y_test)
target_names = target_encoder.inverse_transform(unique_classes)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, labels=unique_classes, target_names = target_names))