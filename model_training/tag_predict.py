import json
import joblib
import pandas as pd
import os
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

# Load trained model and preprocessing tools
def load_model():
    try:
        model = joblib.load("models/html_tag_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        tag_encoder = joblib.load("models/tag_encoder.pkl")
        label_encoders = joblib.load("models/label_encoders.pkl")
        return model, scaler, tag_encoder, label_encoders
    except FileNotFoundError as e:
        print(f"Error: Model file missing - {e}")
        exit(1)

def extract_features(node, depth=0, parent_tag=None, sibling_count=0):
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
        features.extend(extract_features(child, depth=depth+1, parent_tag=node_type, sibling_count=len(children)-1))
    
    return features

# Preprocess extracted features
def preprocess_features(features, scaler, label_encoders):
    df = pd.DataFrame(features)
    
# Define column categories based on the dataset attributes
    categorical_cols = ["type", "parent_tag","characters", "font_name"]  # Adjust as needed
    numerical_cols = ['width', 'height', 'has_text', 'depth', 'num_children', 'sibling_count', 'is_leaf', 'font_size', 'has_font_size',
                       'has_text_color', 'color_r', 'color_g', 'color_b', 'has_background_color', 'background_r', 'background_g',
                       'background_b', 'border_radius', 'border_r', 'border_g', 'border_b', 'border_weight',
                       'has_shadow', 'shadow_r', 'shadow_g', 'shadow_b','shadow_radius', 'text_length', 'word_count', 'contains_number', 'contains_special_chars', 'has_border', 'border_opacity', 'x_quarter', 'y_quarter', 'aspect_ratio', 'area',
                       'normalized_width', 'normalized_height']

    df.fillna(0, inplace=True)

    # Convert categorical features
    for col in categorical_cols:
        if col in label_encoders and df[col].iloc[0] in label_encoders[col].classes_:
            df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        else:
            df[col] = -1  # Assign default if unseen
    
    # Scale numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df

# Predict HTML tag
def predict_tag(features, model, scaler, tag_encoder, label_encoders):
    df = preprocess_features(features, scaler, label_encoders)
    pred = model.predict(df)
    return tag_encoder.inverse_transform(pred)[0]

# Update JSON with predicted tags
def update_json_tags(node, model, scaler, tag_encoder, label_encoders, depth=0, parent_tag=None, sibling_count=0):
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
            df['x_quarter'] = None
            df['y_quarter'] = None
        
        df['aspect_ratio'] = df.apply(
            lambda row: row['width'] / row['height'] if row['height'] and row['height'] != 0 else None, axis=1
        )
        df['area'] = df['width'] * df['height']
        if total_width:
            df['normalized_width'] = df['width'] / total_width
        else:
            df['normalized_width'] = None
        if total_height:
            df['normalized_height'] = df['height'] / total_height
        else:
            df['normalized_height'] = None


        df = df.drop(columns=['x'])
        df = df.drop(columns=['y'])
        df = df.drop(columns=['x_normalized'])
        df = df.drop(columns=['y_normalized'])
        df = df.drop(columns=['x_center'])
        df = df.drop(columns=['y_center'])
        df = df.drop(columns=['tag'])

        # Append this batch to the CSV file


        # Compute min and max for each column
        min_max_values = {col: (df[col].min(), df[col].max()) for col in normalize_columns}

        # Apply Min-Max normalization (scaling between 0 and 1)
        for col in normalize_columns:
            min_val, max_val = min_max_values[col]
            if max_val > min_val:  # Avoid division by zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0  # If min and max are the same, set to 0

        node["tag"] = predict_tag(df, model, scaler, tag_encoder, label_encoders)
    
    for child in node.get("children", []) or []:
        update_json_tags(child, model, scaler, tag_encoder, label_encoders, depth + 1, node["tag"], len(node.get("children", [])) - 1)
    
    return node

# Process JSON file
def process_json_file(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    model, scaler, tag_encoder, label_encoders = load_model()
    updated_data = update_json_tags(data, model, scaler, tag_encoder, label_encoders)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(updated_data, file, indent=4)

# Example usage
input_json = "test_json/input.json"
output_json = "test_json/output.json"
process_json_file(input_json, output_json)
print(f"Updated JSON saved to {output_json}")
