import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import math
import pandas as pd

def load_model_and_encoders():
    """Load trained model and preprocessing encoders."""
    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")
    
    # Load OneHotEncoder
    ohe = joblib.load("ohe_encoder.pkl")
    
    # Load imputer and scaler
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Define model architecture (must match the training script)
    class ImprovedTagClassifier(nn.Module):
        def __init__(self, input_size, output_size, dropout_rate=0.3):
            super(ImprovedTagClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, output_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.dropout(self.relu(self.bn1(self.fc1(x))))
            x = self.dropout(self.relu(self.bn2(self.fc2(x))))
            x = self.dropout(self.relu(self.bn3(self.fc3(x))))
            logits = self.fc4(x)
            return logits

    # Load model
    model = ImprovedTagClassifier(
        input_size=ohe.get_feature_names_out().shape[0] + imputer.statistics_.shape[0],
        output_size=len(label_encoder.classes_)
    )
    model.load_state_dict(torch.load("best_tag_classifier.pth", map_location=torch.device('cpu'),weights_only=True))
    model.eval()
    
    return model, label_encoder, ohe, imputer, scaler

def color_difference(color1, color2):
    """
    Calculate a perceptual color difference between two RGB colors using 
    a simplified version of the Delta E formula.
    Returns a value between 0 and 1, where 0 means identical and 1 means completely different.
    """
    if not all([color1, color2]):
        return 0
    
    # Extract RGB values
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    # Calculate Euclidean distance in RGB space (simplified)
    distance = math.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
    
    # Normalize to 0-1 range (max possible distance is sqrt(3 * 255^2))
    max_distance = math.sqrt(3 * 255**2)
    normalized_distance = distance / max_distance
    
    return normalized_distance

def extract_features(node, sibling_count=0, prev_sibling_tag=None,parent_height=0, parent_bg_color=None):
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))

    text = node_data.get("characters", "")
    text_length = len(text)
    
    children = node.get("children", [])
    num_direct_children = len(children)
    
    # Initialize child tag features
    child_1_tag = None
    child_2_tag = None
    child_3_tag = None
    child_1_percent = 0
    child_2_percent = 0
    child_3_percent = 0
    
    # Calculate node area
    node_width = node_data.get("width", 0)
    node_height = node_data.get("height", 0)
    node_area = node_width * node_height
    
    # Extract child information if available
    if num_direct_children > 0:
        # Child 1
        if len(children) >= 1:
            child_1_tag = children[0].get("tag", "")
            child_1_width = children[0].get("node", {}).get("width", 0)
            child_1_height = children[0].get("node", {}).get("height", 0)
            child_1_area = child_1_width * child_1_height
            child_1_percent = (child_1_area / node_area) if node_area > 0 else 0
        
        # Child 2
        if len(children) >= 2:
            child_2_tag = children[1].get("tag", "")
            child_2_width = children[1].get("node", {}).get("width", 0)
            child_2_height = children[1].get("node", {}).get("height", 0)
            child_2_area = child_2_width * child_2_height
            child_2_percent = (child_2_area / node_area) if node_area > 0 else 0
        
        # Child 3
        if len(children) >= 3:
            child_3_tag = children[2].get("tag", "")
            child_3_width = children[2].get("node", {}).get("width", 0)
            child_3_height = children[2].get("node", {}).get("height", 0)
            child_3_area = child_3_width * child_3_height
            child_3_percent = (child_3_area / node_area) if node_area > 0 else 0
    
    # Count all children in the subtree (recursive count)
    def count_all_descendants(node):
        count = 0
        for child in node.get("children", []):
            # Count this child
            count += 1
            # Add all its descendants
            count += count_all_descendants(child)
        return count
    
    # Count chars to the end
    def count_chars_to_end(node):
        count = 0
        for child in node.get("children", []):
            # Count this child
            node_data = child.get("node", {})
            count += len(node_data.get("characters", ""))
            # Add all its descendants
            count += count_chars_to_end(child)
        return count
    
    # Calculate total descendants
    num_children_to_end = count_all_descendants(node)
    chars_count_to_end = count_chars_to_end(node)
    bg_color = None
    
    # type,width,height,num_direct_children,num_children_to_end,sibling_count,prev_sibling_html_tag,has_background_color,border_radius,has_border,text_length,chars_count_to_end,aspect_ratio,child_1_html_tag,child_2_html_tag,child_3_html_tag,child_1_percentage_of_parent,child_2_percentage_of_parent,child_3_percentage_of_parent,distinct_background
    
    feature = {
        "type": node_type,
        "width": node_width,
        "height": node_height/(parent_height if parent_height else node_height if node_height else 1),
        "num_direct_children": num_direct_children,
        "num_children_to_end": num_children_to_end,  # Total descendants count
        "sibling_count": sibling_count,
        "prev_sibling_html_tag": prev_sibling_tag if prev_sibling_tag else "",
        "has_background_color": 0,
        "border_radius": 0,
        "has_border": 0,
        "text_length": text_length,
        "chars_count_to_end": chars_count_to_end,
        "aspect_ratio": node_width / node_height if node_height > 0 else 0,
        "child_1_html_tag": child_1_tag,
        "child_2_html_tag": child_2_tag,
        "child_3_html_tag": child_3_tag,
        "child_1_percentage_of_parent": child_1_percent,
        "child_2_percentage_of_parent": child_2_percent,
        "child_3_percentage_of_parent": child_3_percent,
        "distinct_background": 0,
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
            feature["has_background_color"] = 1  # Flag for explicit background color
            bg_color = (r, g, b)
            if parent_bg_color:
                bg_difference = color_difference(bg_color, parent_bg_color)
                # If difference is significant (threshold of 0.3 - adjust as needed)
                if bg_difference > 0.4:
                    feature["distinct_background"] = 1   
            break
    # Also check backgrounds for background color
    backgrounds = node_data.get("backgrounds", [])
    for bg in backgrounds:
        if bg.get("type") == "SOLID" and "color" in bg:
            r, g, b = (
                int(bg["color"].get("r", 0) * 255),
                int(bg["color"].get("g", 0) * 255),
                int(bg["color"].get("b", 0) * 255),
            )
            feature["has_background_color"] = 1  # Flag for explicit background color
            bg_color = (r, g, b)
            if parent_bg_color:
                bg_difference = color_difference(bg_color, parent_bg_color)
                # If difference is significant (threshold of 0.3 - adjust as needed)
                if bg_difference > 0.4:
                    feature["distinct_background"] = 1   
            break
    
    # Extract strokes (borders)
    strokes = node_data.get("strokes", [])
    if strokes:
        feature["has_border"] = 1
    
    # Extract border radius
    br_top_left = node_data.get("topLeftRadius", 0)
    br_top_right = node_data.get("topRightRadius", 0)
    br_bottom_left = node_data.get("bottomLeftRadius", 0)
    br_bottom_right = node_data.get("bottomRightRadius", 0)
    
    if any([br_top_left, br_top_right, br_bottom_left, br_bottom_right]):
        feature["border_radius"] = (br_top_left + br_top_right + br_bottom_left + br_bottom_right) / 4
        if feature["border_radius"] >= 50:
            feature["border_radius"] = 0
    
    return feature

def predict_tag(node,sibling_count,prev_sibling_tag,parent_height,parent_bg_color, model, label_encoder, ohe, imputer, scaler):
    """
    Predict tag for a node using the trained model.
    """
    # First, recursively process children to get their tags and information
    node_data = node.get("node", {})
    figma_type = node_data.get("type","")
    node_height = node_data.get("height", 0)
    children = node.get("children", [])
    
    # Extract bgcolor
    fills = node_data.get("fills", [])
    has_background_color = False
    bg_color = None
    for fill in fills:
        if fill.get("type") == "SOLID" and "color" in fill:
            r, g, b = (
                int(fill["color"].get("r", 0) * 255),
                int(fill["color"].get("g", 0) * 255),
                int(fill["color"].get("b", 0) * 255),
            )
            has_background_color = True
            bg_color = (r, g, b)                
            break
    backgrounds = node_data.get("backgrounds", [])
    for bg in backgrounds:
        if bg.get("type") == "SOLID" and "color" in bg:
            r, g, b = (
                int(bg["color"].get("r", 0) * 255),
                int(bg["color"].get("g", 0) * 255),
                int(bg["color"].get("b", 0) * 255),
            )
            has_background_color = True
            bg_color = (r, g, b)    
            break
    
    prev_sib_tag = None
    # Process children first - left to right
    for i, child in enumerate(children):
        # Predict child tag
        predict_tag(child,len(children)-1,prev_sib_tag,node_height,bg_color if has_background_color else parent_bg_color, model, label_encoder, ohe, imputer, scaler)
        
        # Update previous sibling tag for next iteration
        prev_sib_tag = child.get("tag","UNK")


    # Extract features after processing children
    feature = extract_features(
        node,
        sibling_count,
        prev_sibling_tag,
        parent_height,
        parent_bg_color,
    )
    
    # handle easy cases here
    #TODO: TEXT,SVG,HR,IMAGE
    if figma_type == "TEXT":
        node["tag"] = "P"
    if figma_type == "SVG":
        node["tag"] = "SVG"
    if figma_type == "LINE":
        node["tag"] = "HR"
    if (fills := node_data.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills): 
        node["tag"] = "IMG"
        
    # If tag is already defined and not 'UNK', return it
    if node.get("tag", "").upper() != "UNK":
        return
    
    # Prepare features for model
    categorical_cols = ['type','prev_sibling_html_tag','child_1_html_tag','child_2_html_tag','child_3_html_tag']
    continuous_cols = [col for col in feature.keys() if col not in categorical_cols]
    
    # Prepare categorical data
    cat_data = [[feature[col] for col in categorical_cols]]
    X_cat_df = pd.DataFrame(cat_data, columns=categorical_cols)  
    cat_encoded = ohe.transform(X_cat_df)
    
    # Prepare continuous data
    cont_data = [[feature.get(col, 0) for col in continuous_cols]]
    X_df = pd.DataFrame(cont_data, columns=continuous_cols)
    cont_imputed = imputer.transform(X_df)
    cont_scaled = scaler.transform(cont_imputed)
    
    # Combine features
    X_processed = np.concatenate([cat_encoded, cont_scaled], axis=1)
    
    # Convert to tensor and predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_tag = label_encoder.inverse_transform(predicted)[0]
    
    # Update the node's tag
    node["tag"] = predicted_tag

def process_figma_json(input_file, output_file):
    """
    Process a Figma JSON file, predicting tags for UNK nodes.
    """
    # Load model and preprocessing tools
    model, label_encoder, ohe, imputer, scaler = load_model_and_encoders()
    
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Predict tags recursively
    predict_tag(data,0,None,None,None, model, label_encoder, ohe, imputer, scaler)
    
    # Save processed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {input_file}. Output saved to {output_file}")

# Example usage
if __name__ == "__main__":
    process_figma_json("input.json", "output.json")