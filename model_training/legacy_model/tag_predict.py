import json
import joblib
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Columns to be normalized
normalize_columns = [
    "area",
    "word_count",
    "text_length",
    "font_size",
    "sibling_count",
    "num_children",
    "height",
    "width",
]

# Define the Neural Network Model
class TagClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(TagClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 128)         # Second hidden layer
        self.fc3 = nn.Linear(128, 256)        # Third hidden layer
        self.fc4 = nn.Linear(256, 512)        # Fourth hidden layer
        self.fc5 = nn.Linear(512, 512)        # Fifth hidden layer
        self.fc6 = nn.Linear(512, 512)        # Sixth hidden layer
        self.fc7 = nn.Linear(512, 256)        # Seventh hidden layer
        self.fc8 = nn.Linear(256, 128)        # Eighth hidden layer
        self.fc9 = nn.Linear(128, output_size) # Output layer
        self.relu = nn.ReLU()                 # Non-linear activation

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        logits = self.fc9(x)  # No activation here: CrossEntropyLoss expects raw logits.
        return logits


# Load trained model and preprocessing tools
def load_model():
    try:
        # Load scaler and encoders
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        # Create empty label_encoders dict - will handle encoding in preprocessing
        label_encoders = {}
        
        # Get model input size from the saved model
        saved_model = torch.load("models/tag_classifier.pth")
        input_size = saved_model['fc1.weight'].shape[1]  # Extract input size from first layer weights
        
        # Create model with correct input size
        output_size = len(label_encoder.classes_)
        model = TagClassifier(input_size, output_size)
        model.load_state_dict(saved_model)
        model.eval()  # Set model to evaluation mode
        
        return model, scaler, label_encoder, label_encoders
    except FileNotFoundError as e:
        print(f"Error: Model file missing - {e}")
        exit(1)


def extract_features(node, depth=0, parent_tag=None, sibling_count=0, parent_tag_html=None):
    features = []
    
    tag = node.get("tag", "")
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))

    text = node_data.get("characters", "")
    text_length = len(text)
    word_count = len(text.split()) if text else 0
    contains_number = any(ch.isdigit() for ch in text)
    contains_special_chars = any(not ch.isalnum() and not ch.isspace() for ch in text)
    
    children = node.get("children", [])
    num_children = len(children)
    is_leaf = 1 if num_children == 0 else 0
    
    feature = {
        "tag": tag,
        "type": node_type,
        "x": node_data.get("x", 0),
        "y": node_data.get("y", 0),
        "width": node_data.get("width", 0),
        "height": node_data.get("height", 0),
        "characters": text,
        "has_text": int(bool(text)),
        "depth": depth,
        "num_children": num_children,
        "parent_tag": parent_tag if parent_tag else "",
        "parent_tag_html": parent_tag_html if parent_tag_html else "",
        "sibling_count": sibling_count,
        "is_leaf": is_leaf,
        "font_size": node_data.get("fontSize", 16),
        "has_font_size": int("fontSize" in node_data),
        "font_name": node_data.get("fontName", {}).get("style", "") if node_data.get("fontName") else "normal",
        "has_text_color": 0, "color_r": 0, "color_g": 0, "color_b": 0,
        "has_background_color": 0, "background_r": 0, "background_g": 0, "background_b": 0,
        "border_radius": 0,
        "border_r": 0, "border_g": 0, "border_b": 0,
        "has_border": 0, "border_opacity": 0,
        "border_weight": node_data.get("strokeWeight", 0),
        "has_shadow": 0, "shadow_r": 0, "shadow_g": 0, "shadow_b": 0,
        "shadow_radius": 0, 
        "text_length": text_length,
        "word_count": word_count,
        "contains_number": int(contains_number),
        "contains_special_chars": int(contains_special_chars),
    }
    
    # Extract fills (background and text color)
    fills = node_data.get("fills", [])
    for fill in fills:
        if fill.get("type") == "SOLID" and "color" in fill:
            r, g, b = (
                int(fill["color"].get("r", 0) * 255),
                int(fill["color"].get("g", 0) * 255),
                int(fill["color"].get("b", 0) * 255),
            )
            feature["color_r"], feature["color_g"], feature["color_b"] = r, g, b
            feature["has_text_color"] = 1  # Flag indicating explicit text color is set
            
            feature["background_r"], feature["background_g"], feature["background_b"] = r, g, b
            feature["has_background_color"] = 1  # Flag for explicit background color
            break  
    
    # Extract strokes (borders)
    strokes = node_data.get("strokes", [])
    if strokes:
        stroke = strokes[0]
        feature["has_border"] = 1
        if "color" in stroke:
            feature["border_r"], feature["border_g"], feature["border_b"] = (
                int(stroke["color"].get("r", 0) * 255),
                int(stroke["color"].get("g", 0) * 255),
                int(stroke["color"].get("b", 0) * 255),
            )
        feature["border_opacity"] = stroke.get("opacity", 0)
    
    # Extract border radius
    br_top_left = node_data.get("topLeftRadius", 0)
    br_top_right = node_data.get("topRightRadius", 0)
    br_bottom_left = node_data.get("bottomLeftRadius", 0)
    br_bottom_right = node_data.get("bottomRightRadius", 0)
    
    if any([br_top_left, br_top_right, br_bottom_left, br_bottom_right]):
        feature["border_radius"] = (br_top_left + br_top_right + br_bottom_left + br_bottom_right) / 4
    
    # Extract shadow
    effects = node_data.get("effects", [])
    for effect in effects:
        if effect.get("type") == "DROP_SHADOW":
            feature["has_shadow"] = 1
            if "color" in effect:
                feature["shadow_r"], feature["shadow_g"], feature["shadow_b"] = (
                    int(effect["color"].get("r", 0) * 255),
                    int(effect["color"].get("g", 0) * 255),
                    int(effect["color"].get("b", 0) * 255),
                )
            feature["shadow_radius"] = effect.get("radius", 0)
            break  
    
    features.append(feature)
    
    for child in children:
        features.extend(extract_features(child, depth=depth+1, parent_tag=node_type, sibling_count=len(children)-1, parent_tag_html= tag))
    
    return features


def preprocess_features(features, scaler, label_encoders):
    df = pd.DataFrame(features)
    
    # Define only categorical columns explicitly
    categorical_cols = ["tag", "type", "parent_tag", "parent_tag_html"]
    
    # All other columns will be treated as numerical
    numerical_cols = [col for col in df.columns if col not in categorical_cols]
    
    df.fillna(0, inplace=True)
    
    # Convert categorical features
    for col in categorical_cols:
        if col in label_encoders and col in df.columns:
            # Check if the value is in the encoder's classes, otherwise use -1
            df[col] = df[col].apply(lambda x:
                                   label_encoders[col].transform([str(x)])[0]
                                   if x is not None and str(x) in label_encoders[col].classes_
                                   else -1)
        else:
            if col in df.columns:
                df[col] = -1  # Assign default if unseen
    
    # Scale numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df
# Predict HTML tag using PyTorch model
def predict_tag(features, model, scaler, label_encoder, label_encoders):
    df = preprocess_features(features, scaler, label_encoders)
    
    # Debug: Print the shape of the DataFrame
    print(f"DataFrame shape: {df.shape}")
    
    # Make sure the model input has the expected number of features
    expected_features = model.fc1.in_features  # This gets the expected input size from your model
    current_features = df.shape[1]
    
    if current_features != expected_features:
        print(f"Warning: Model expects {expected_features} features but got {current_features}")
        
        # Option 1: Add missing columns with zeros
        if current_features < expected_features:
            missing_features = expected_features - current_features
            for i in range(missing_features):
                col_name = f"missing_feature_{i}"
                df[col_name] = 0
        
        # Option 2: If too many features, select only the ones the model knows about
        elif current_features > expected_features:
            # This assumes you know which features the model was trained on
            # You might need to adjust this based on your specific case
            print("Too many features - trimming to expected number")
            # Keep only the first expected_features columns
            df = df.iloc[:, :expected_features]
    
    # Convert DataFrame to PyTorch tensor
    input_tensor = torch.tensor(df.values, dtype=torch.float32)
    
    # Debug: Print tensor shape
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Convert prediction back to tag name
    return label_encoder.inverse_transform(predicted.numpy())[0]

# Update JSON with predicted tags
def update_json_tags(node, model, scaler, label_encoder, label_encoders, depth=0, parent_tag=None, sibling_count=0):
    if node.get("tag") == "UNK":
        features_list = extract_features(node, depth, parent_tag, sibling_count)
        df = pd.DataFrame(features_list)
        
        # Normalize positions per JSON file to avoid cross-file influence
        min_x = df['x'].min() if df['x'].notnull().any() else 0
        min_y = df['y'].min() if df['y'].notnull().any() else 0
        df['x_normalized'] = df['x'] - min_x
        df['y_normalized'] = df['y'] - min_y
        
        df['x_center'] = df['x'] + df['width'] / 2
        df['y_center'] = df['y'] + df['height'] / 2
        
        # Attempt to compute total dimensions using a BODY tag if available
        body_node = df[df['tag'] == 'BODY']
        if not body_node.empty:
            total_width = body_node.iloc[0]['width']
            total_height = body_node.iloc[0]['height']
        else:
            total_width = (df['x'] + df['width']).max()
            total_height = (df['y'] + df['height']).max()
        
        # Avoid division by zero
        if total_width and total_height:
            df['x_quarter'] = df['x_center'] / total_width
            df['y_quarter'] = df['y_center'] / total_height
        else:
            df['x_quarter'] = 0
            df['y_quarter'] = 0
        
        df['aspect_ratio'] = df.apply(
            lambda row: row['width'] / row['height'] if row['height'] and row['height'] != 0 else 0, axis=1
        )
        df['area'] = df['width'] * df['height']
        if total_width:
            df['normalized_width'] = df['width'] / total_width
        else:
            df['normalized_width'] = 0
        if total_height:
            df['normalized_height'] = df['height'] / total_height
        else:
            df['normalized_height'] = 0

        # Drop columns not needed for prediction
        df = df.drop(columns=['x'])
        df = df.drop(columns=['y'])
        df = df.drop(columns=['x_normalized'])
        df = df.drop(columns=['y_normalized'])
        df = df.drop(columns=['x_center'])
        df = df.drop(columns=['y_center'])
        df = df.drop(columns=['characters'])
        df = df.drop(columns=['font_size'])
        df = df.drop(columns=['font_name'])
        df = df.drop(columns=['color_r'])
        df = df.drop(columns=['color_g'])
        df = df.drop(columns=['color_b'])
        df = df.drop(columns=['background_r'])
        df = df.drop(columns=['background_g'])
        df = df.drop(columns=['background_b'])
        df = df.drop(columns=['border_radius'])
        df = df.drop(columns=['border_r'])
        df = df.drop(columns=['border_g'])
        df = df.drop(columns=['border_b'])
        df = df.drop(columns=['border_opacity'])
        df = df.drop(columns=['border_weight'])
        df = df.drop(columns=['shadow_r'])
        df = df.drop(columns=['shadow_g'])
        df = df.drop(columns=['shadow_b'])
        df = df.drop(columns=['shadow_radius'])
        df = df.drop(columns=['word_count'])
        df = df.drop(columns=['normalized_width'])
        df = df.drop(columns=['normalized_height'])
        df = df.drop(columns=['contains_special_chars'])
        df = df.drop(columns=['contains_number'])
        df = df.drop(columns=['has_shadow'])
        df = df.drop(columns=['has_border'])
        df = df.drop(columns=['has_text_color'])
        df = df.drop(columns=['height'])
        df = df.drop(columns=['has_text'])
        df = df.drop(columns=['depth'])
        df = df.drop(columns=['x_quarter'])
        df = df.drop(columns=['y_quarter'])
        df = df.drop(columns=['area'])
        df = df.drop(columns=['has_font_size'])

        # Apply Min-Max normalization for specific columns
        min_max_values = {col: (df[col].min(), df[col].max()) for col in normalize_columns if col in df.columns}
        for col in normalize_columns:
            if col in df.columns:
                min_val, max_val = min_max_values[col]
                if max_val > min_val:  # Avoid division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0  # If min and max are the same, set to 0

        # Use the model to predict the tag
        predicted_tag = predict_tag(df.to_dict('records'), model, scaler, label_encoder, label_encoders)
        node["tag"] = predicted_tag
    
    # Recursively process children
    for child in node.get("children", []) or []:
        update_json_tags(child, model, scaler, label_encoder, label_encoders, depth + 1, node["tag"], len(node.get("children", [])) - 1)
    
    return node

# Process JSON file
def process_json_file(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    model, scaler, label_encoder, label_encoders = load_model()
    updated_data = update_json_tags(data, model, scaler, label_encoder, label_encoders)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(updated_data, file, indent=4)

# Example usage
if __name__ == "__main__":
    input_json = "test_json/input.json"
    output_json = "test_json/output.json"
    process_json_file(input_json, output_json)
    print(f"Updated JSON saved to {output_json}")