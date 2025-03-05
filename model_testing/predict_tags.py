import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import math
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
body_width = None
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_model_and_encoders():
    """Load trained model and preprocessing encoders."""
    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")
    
    # Load OneHotEncoder
    ohe = joblib.load("ohe_encoder.pkl")
    
    # Load imputer and scaler
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    scaler_emb = joblib.load("scaler_emb.pkl")
    
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
        input_size=ohe.get_feature_names_out().shape[0] + imputer.statistics_.shape[0] + 384,
        output_size=len(label_encoder.classes_)
    )
    model.load_state_dict(torch.load("best_tag_classifier.pth", map_location=torch.device('cpu'),weights_only=True))
    model.eval()
    
    return model, label_encoder, ohe, imputer, scaler,scaler_emb

def extract_semantic_features(node, text_nodes_with_content):
    """
    Calculate the semantic features from the nearest text node.
    
    Args:
    node (dict): Current node being processed
    text_nodes_with_content (list): List of text nodes with their x, y coordinates and text content
    
    Returns:
    dict: Semantic feature dictionary
    """
    if not text_nodes_with_content:
        return {
            "nearest_text_semantic_vector": [0] * 384,  # Default zero vector
            "nearest_text_semantic_distance": 9999999
        }
    
    # Get current node's center coordinates
    node_data = node.get("node", {})
    x = node_data.get("x", 0) + node_data.get("width", 0) / 2
    y = node_data.get("y", 0) + node_data.get("height", 0) / 2
    
    # Calculate Euclidean distances to all text nodes and find the nearest
    min_distance = float('inf')
    nearest_text_node = None
    
    for text_node in text_nodes_with_content:
        tx, ty = text_node['x'], text_node['y']
        distance = math.sqrt((x - tx)**2 + (y - ty)**2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_text_node = text_node
    
    # If a nearest text node is found, get its semantic embedding
    if nearest_text_node and nearest_text_node.get('text'):
        # Generate semantic embedding for the text
        semantic_vector = semantic_model.encode(nearest_text_node['text'])
        
        return {
            "nearest_text_semantic_vector": semantic_vector.tolist(),  # Convert to list for JSON serialization
            "nearest_text_semantic_distance": min_distance
        }
    
    # Fallback if no meaningful text is found
    return {
        "nearest_text_semantic_vector": [0] * 384,  # Default zero vector
        "nearest_text_semantic_distance": 9999999
    }

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

def extract_features(node, sibling_count=0, prev_sibling_tag=None,parent_height=0, parent_bg_color=None,text_nodes=None):
    global body_width
    
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
    
    # type,width,height,num_direct_children,num_children_to_end,sibling_count,prev_sibling_html_tag,has_background_color,border_radius,has_border,text_length,chars_count_to_end,aspect_ratio,child_1_html_tag,child_2_html_tag,child_3_html_tag,child_1_percentage_of_parent,child_2_percentage_of_parent,child_3_percentage_of_parent,distinct_background,nearest_text_node_dist
    
    feature = {
        "type": node_type,
        "width": node_width/(body_width if body_width else 1),
        "height": node_height/(parent_height if parent_height else node_height if node_height else 1),
        "num_children_to_end": num_children_to_end,  # Total descendants count
        "sibling_count": sibling_count,
        "prev_sibling_html_tag": prev_sibling_tag if prev_sibling_tag else "",
        "has_background_color": 0,
        "border_radius": 0,
        "text_length": text_length,
        "chars_count_to_end": chars_count_to_end,
        "aspect_ratio": node_width / node_height if node_height > 0 else 0,
        "child_1_html_tag": child_1_tag,
        "child_2_html_tag": child_2_tag,
        "child_1_percentage_of_parent": child_1_percent,
        "child_2_percentage_of_parent": child_2_percent,
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
                if bg_difference > 0.25:
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
            a = min(float(fill["color"].get("a", 1)),float(fill.get("opacity",1)))
            
            bg_color = (r*a, g*a, b*a)
            
            if parent_bg_color:
                bg_difference = color_difference(bg_color, parent_bg_color)
                                
                # If difference is significant (threshold of 0.3 - adjust as needed)
                if bg_difference > 0.2:
                    feature["distinct_background"] = 1  
                
            break
    
    # Extract border radius
    br_top_left = node_data.get("topLeftRadius", 0)
    br_top_right = node_data.get("topRightRadius", 0)
    br_bottom_left = node_data.get("bottomLeftRadius", 0)
    br_bottom_right = node_data.get("bottomRightRadius", 0)
    
    if any([br_top_left, br_top_right, br_bottom_left, br_bottom_right]):
        feature["border_radius"] = (br_top_left + br_top_right + br_bottom_left + br_bottom_right) / 4
        if feature["border_radius"] >= 50:
            feature["border_radius"] = 0
            
    # Get semantic features for the current node
    semantic_features = extract_semantic_features(node, text_nodes)
    
    # Calculate nearest text node distance
    nearest_text_distance = semantic_features.get('nearest_text_semantic_distance',0)
    nearest_text_semantic = semantic_features.get('nearest_text_semantic_vector',[0]*384)
    
    # Add nearest text node distance to the feature dictionary
    feature["nearest_text_node_dist"] = (nearest_text_distance) / (math.sqrt((node_width)* (node_height)) if math.sqrt((node_width)*(node_height)) else 1)
    feature["nearest_text_semantic"] = nearest_text_semantic
    
    return feature

def predict_tag(node,sibling_count,prev_sibling_tag,parent_height,parent_bg_color,text_nodes, model, label_encoder, ohe, imputer, scaler, scaler_emb):
    """
    Predict tag for a node using the trained model.
    """
    
    global body_width
    
    # First pass: Collect text nodes if not provided
    if text_nodes is None:
        def collect_text_nodes(node):
            text_nodes_list = []
            # Function to check if a node has meaningful text
            def has_meaningful_text(node_data):
                return node_data.get('type','') == "TEXT"
            
            node_data = node.get("node", {})
            # If this node has meaningful text
            if has_meaningful_text(node_data):
                text_nodes_list.append({
                    'x': node_data.get("x", 0) + node_data.get("width", 0) / 2,
                    'y': node_data.get("y", 0) + node_data.get("height", 0) / 2
                })
            
            # Recursively check children
            for child in node.get("children", []):
                text_nodes_list.extend(collect_text_nodes(child))
            
            return text_nodes_list
        
        text_nodes = collect_text_nodes(node)
    
    # First, recursively process children to get their tags and information
    node_data = node.get("node", {})
    figma_type = node_data.get("type","")
    node_height = node_data.get("height", 0)
    if not body_width or (body_width and body_width == 0):
        body_width = node_data.get("width", 0)
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
        predict_tag(child,len(children)-1,prev_sib_tag,node_height,bg_color if has_background_color else parent_bg_color,text_nodes, model, label_encoder, ohe, imputer, scaler, scaler_emb)
        
        # Update previous sibling tag for next iteration
        prev_sib_tag = child.get("tag","UNK")


    # Extract features after processing children
    feature = extract_features(
        node,
        sibling_count,
        prev_sibling_tag,
        parent_height,
        parent_bg_color,
        text_nodes
    )
    
    # handle easy cases here
    #TODO: TEXT,SVG,HR,IMAGE,VECTOR
    if figma_type == "TEXT":
        node["tag"] = "P"
    if figma_type == "SVG" or figma_type == "VECTOR":
        node["tag"] = "SVG"
    if figma_type == "LINE":
        node["tag"] = "HR"
    if (fills := node_data.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills): 
        node["tag"] = "IMG"
        
    # If tag is already defined and not 'UNK', return it
    if node.get("tag", "").upper() != "UNK":
        return
    
    # Prepare features for model
    categorical_cols = ['type','prev_sibling_html_tag','child_1_html_tag','child_2_html_tag']
    continuous_cols = [col for col in feature.keys() if col not in categorical_cols and col != 'nearest_text_semantic']
    
    # Handle nearest_text_semantic column safely
    nearest_text_semantic = feature.get('nearest_text_semantic_vector',[0]*384)
    nearest_text_semantic_expanded = np.array(nearest_text_semantic, dtype=float).reshape(1, -1)
    
    # Prepare categorical data
    cat_data = [[feature[col] for col in categorical_cols]]
    X_cat_df = pd.DataFrame(cat_data, columns=categorical_cols)  
    cat_encoded = ohe.transform(X_cat_df)
    
    # Prepare continuous data
    cont_data = [[feature.get(col, 0) for col in continuous_cols]]
    X_df = pd.DataFrame(cont_data, columns=continuous_cols)
    cont_imputed = imputer.transform(X_df)
    cont_scaled = scaler.transform(cont_imputed)
    nearest_text_semantic_scaled = scaler_emb.transform(nearest_text_semantic_expanded)
    
    # Combine features
    X_processed = np.concatenate([cat_encoded, cont_scaled, nearest_text_semantic_scaled], axis=1)
    
    # Convert to tensor and predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_tag = label_encoder.inverse_transform(predicted)[0]
    
    # Save data to CSV
    X_cat_df["predicted_tag"] = predicted_tag  # Add predicted tag to categorical features
    X_full_df = pd.concat([X_cat_df, X_df], axis=1)  # Combine categorical and continuous features
    X_full_df["predicted_tag"] = predicted_tag  # Add predicted tag at the end

    # Save to CSV
    X_full_df.to_csv("features_with_prediction.csv",mode='a', index=False)

    print("Feature vector and predicted tag saved to predicted_features.csv")
    
    # Update the node's tag
    node["tag"] = predicted_tag

def process_figma_json(input_file, output_file):
    """
    Process a Figma JSON file, predicting tags for UNK nodes.
    """
    # Load model and preprocessing tools
    model, label_encoder, ohe, imputer, scaler, scaler_emb = load_model_and_encoders()
    
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if os.path.exists('features_with_prediction.csv'):
        os.remove('features_with_prediction.csv')
        
    # Predict tags recursively
    predict_tag(data,0,None,None,None,None, model, label_encoder, ohe, imputer, scaler,scaler_emb)
    
    # Save processed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {input_file}. Output saved to {output_file}")

# Example usage
if __name__ == "__main__":
    process_figma_json("input.json", "output.json")