import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import math
import pandas as pd
import os
import spacy
import random
import shutil
from lxml import etree
import webbrowser

body_width = None
num_nodes = None
num_chars = None

# Load the pretrained spaCy model
nlp = spacy.load("en_core_web_sm")

def verb_ratio(text):
    doc = nlp(text)
    if len(doc) > 5:
        return 0
    
    verb_count = sum(1 for token in doc if token.pos_ == "VERB" and token.lemma_.lower() not in ["username"])
    total_words = sum(1 for token in doc if token.is_alpha)  # Count only valid words (ignore punctuation)
    
    return verb_count / total_words if total_words > 0 else 0  # Avoid division by zero

def is_near_gray(r, g, b, threshold=30, min_val=50, max_val=200):
    """Check if (r, g, b) is near a shade of gray within a threshold, excluding very dark or very light grays."""
    return (
        min_val <= r <= max_val and
        min_val <= g <= max_val and
        min_val <= b <= max_val and
        abs(r - g) <= threshold and
        abs(g - b) <= threshold and
        abs(r - b) <= threshold
    )

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

def find_nearest_text_node(node, text_nodes):
    """
    Calculate the distance to the nearest text node.
    
    Args:
    node (dict): Current node being processed
    text_nodes (list): List of text nodes with their x, y coordinates
    
    Returns:
    float: Distance to the nearest text node, or a large value if no text nodes exist
    """
    if not text_nodes:
        return 9999999  # Large default value if no text nodes exist
    
    # Get current node's center coordinates
    node_data = node.get("node", {})
    x = node_data.get("x", 0) + node_data.get("width", 0) / 2
    y = node_data.get("y", 0) + node_data.get("height", 0) / 2
    
    # Calculate Euclidean distances to all text nodes
    min_distance = float('inf')
    for text_node in text_nodes:
        tx, ty = text_node['x'], text_node['y']
        distance = math.sqrt((x - tx)**2 + (y - ty)**2)
        min_distance = min(min_distance, distance)
    
    return min_distance

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
    global num_nodes
    global num_chars
    
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))

    text = node_data.get("characters", "")
    
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
    
    has_placeholder = 0
    is_verb = 0
    # Extract child information if available
    if num_direct_children > 0:
        # Child 1
        if len(children) >= 1:
            child_1_tag = children[0].get("tag", "")
            child_1_type = children[0].get("node",{}).get("type", "")
            if child_1_type == "TEXT":
                if is_verb == 0:
                    is_verb = verb_ratio(children[0].get("node", {}).get("characters", ""))
                placeholder_fills = children[0].get("node", {}).get("fills", [])
                for fill in placeholder_fills:
                    if fill.get("type") == "SOLID" and "color" in fill:
                        r, g, b = (
                            int(fill["color"].get("r", 0) * 255),
                            int(fill["color"].get("g", 0) * 255),
                            int(fill["color"].get("b", 0) * 255),
                        )
                        print(r,g,b)
                        if is_near_gray(r, g, b):
                            has_placeholder = 1
                        break
            child_1_width = children[0].get("node", {}).get("width", 0)
            child_1_height = children[0].get("node", {}).get("height", 0)
            child_1_area = child_1_width * child_1_height
            child_1_percent = (child_1_area / node_area) if node_area > 0 else 0
        
        # Child 2
        if len(children) >= 2:
            child_2_tag = children[1].get("tag", "")
            child_2_type = children[1].get("node",{}).get("type", "")
            if child_2_type == "TEXT" and is_verb == 0:
                is_verb = verb_ratio(children[1].get("node", {}).get("characters", ""))
            child_2_width = children[1].get("node", {}).get("width", 0)
            child_2_height = children[1].get("node", {}).get("height", 0)
            child_2_area = child_2_width * child_2_height
            child_2_percent = (child_2_area / node_area) if node_area > 0 else 0
        
        # Child 3
        if len(children) >= 3:
            child_3_tag = children[2].get("tag", "")
            child_3_type = children[2].get("node",{}).get("type", "")
            if child_3_type == "TEXT" and is_verb == 0:
                is_verb = verb_ratio(children[2].get("node", {}).get("characters", ""))
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
    
    # get center of weight
    def get_center_of_weight(node):
        parent_node_data = node.get("node", {})
        parent_x_center = parent_node_data.get("x", 0) + parent_node_data.get("width", 0) / 2
        
        total_area = 0
        total = 0

        for child in node.get("children", []):
            child_node_data = child.get("node", {})
            x = child_node_data.get("x", 0)
            width = child_node_data.get("width", 0)
            height = child_node_data.get("height", 0)

            child_x_center = x + width / 2
            area = width * height  # Area as weight

            total += area * child_x_center
            total_area += area

        # Compute weighted center
        weighted_x = total / total_area if total_area else parent_x_center

        # Calculate normalized difference
        diff = abs(parent_x_center - weighted_x) / (parent_node_data.get("width", 0) if parent_node_data.get("width", 0) else 1)
        
        return diff
    
    # Calculate total descendants
    num_children_to_end = count_all_descendants(node)
    if not num_nodes or num_nodes == 0:
        num_nodes = num_children_to_end
    chars_count_to_end = count_chars_to_end(node)
    if not num_chars or num_chars == 0:
        num_chars = chars_count_to_end
    bg_color = None
    
    # type,width,height,num_direct_children,num_children_to_end,sibling_count,prev_sibling_html_tag,has_background_color,border_radius,has_border,text_length,chars_count_to_end,aspect_ratio,child_1_html_tag,child_2_html_tag,child_3_html_tag,child_1_percentage_of_parent,child_2_percentage_of_parent,child_3_percentage_of_parent,distinct_background,nearest_text_node_dist
    
    feature = {
        "type": node_type,
        "width": node_width/(body_width if body_width else 1),
        "height": node_height/(parent_height if parent_height else node_height if node_height else 1),
        # "num_children_to_end": num_children_to_end/(num_nodes if num_nodes else 1),  # Total descendants count
        "sibling_count": sibling_count,
        "prev_sibling_html_tag": prev_sibling_tag if prev_sibling_tag else "",
        "has_background_color": 0,
        "border_radius": 0,
        # "chars_count_to_end": chars_count_to_end/(num_chars if num_chars else 1),
        "aspect_ratio": node_width / node_height if node_height > 0 else 0,
        "child_1_html_tag": child_1_tag,
        "child_2_html_tag": child_2_tag,
        "child_1_percentage_of_parent": child_1_percent,
        "child_2_percentage_of_parent": child_2_percent,
        "distinct_background": 0,
        "center_of_weight_diff": get_center_of_weight(node),
        "is_verb": is_verb,
        "has_placeholder": has_placeholder
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
            
    # Calculate nearest text node distance
    nearest_text_distance = find_nearest_text_node(node, text_nodes)
    
    # Add nearest text node distance to the feature dictionary
    feature["nearest_text_node_dist"] = (nearest_text_distance+0.01) / (math.sqrt((node_width+0.001)* (node_height+0.001)) if math.sqrt((node_width+0.001)*(node_height+0.001)) else 1)
    
    return feature

def predict_tag(node,sibling_count,prev_sibling_tag,parent_height,parent_bg_color,text_nodes, model, label_encoder, ohe, imputer, scaler):
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
        predict_tag(child,len(children)-1,prev_sib_tag,node_height,bg_color if has_background_color else parent_bg_color,text_nodes, model, label_encoder, ohe, imputer, scaler)
        
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
    #TODO: GROUP,TEXT,SVG,HR,IMAGE,VECTOR,CHECKBOX,RADIO
    if figma_type == "GROUP":
        node["tag"] = "DIV"
    if figma_type == "TEXT":
        node["tag"] = "P"
    if figma_type == "SVG" or figma_type == "VECTOR":
        node["tag"] = "SVG"
    if figma_type == "LINE":
        node["tag"] = "HR"
    if (fills := node_data.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills): 
        node["tag"] = "IMG"
    if node.get("node", {}).get("width", 0) == node.get("node", {}).get("height", 0) and node.get("node", {}).get("width", 0) < 50:
        strokes = node_data.get("strokes", [])
        fills = node_data.get("fills", [])

        has_solid_fill = any(fill.get("type") == "SOLID" for fill in fills)

        # Extract stroke and fill colors
        stroke_color = strokes[0].get("color") if strokes else None
        fill_color = fills[0].get("color") if fills else None

        if not strokes or (stroke_color == fill_color):
            node["tag"] = "LI"
        elif has_solid_fill:
            if node.get("node", {}).get("type", "RECTANGLE") == "RECTANGLE":
                node["tag"] = "CHECKBOX"
            elif node.get("node", {}).get("type", "ELLIPSE") == "ELLIPSE":
                node["tag"] = "RADIO"
        
    # If tag is already defined and not 'UNK', return it
    if node.get("tag", "").upper() != "UNK":
        return
    
    # Prepare features for model
    categorical_cols = ['type','prev_sibling_html_tag','child_1_html_tag','child_2_html_tag']
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
    
    # print(X_cat_df,X_df)
    
    
    # Combine features
    X_processed = np.concatenate([cat_encoded, cont_scaled], axis=1)
    
    # Convert to tensor and predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_tag = label_encoder.inverse_transform(predicted)[0]
        # node_width = node_data.get("width", 0)
        # node_height = node_data.get("height", 0)
        # aspect_ratio = node_width / node_height if node_height > 0 else 0
        # if predicted_tag == "INPUT" and aspect_ratio < 5:
        #     predicted_tag = "BUTTON"
    # Save data to CSV
    X_cat_df["predicted_tag"] = predicted_tag  # Add predicted tag to categorical features
    X_full_df = pd.concat([X_cat_df, X_df], axis=1)  # Combine categorical and continuous features
    X_full_df["predicted_tag"] = predicted_tag  # Add predicted tag at the end
    print("Predicted tag:",predicted_tag)

    # Save to CSV
    X_full_df.to_csv("features_with_prediction.csv",mode='a', index=False)

    print("Feature vector and predicted tag saved to predicted_features.csv")
    
    # Update the node's tag
    node["tag"] = predicted_tag

def post_process_tags(nodes):
    global body_width
    
    if not isinstance(nodes, dict):
        raise ValueError("Expected a dict with 'children' key but got a different structure")
    
    # Process the children of the root node
    process_nodes(nodes)
    
    return nodes

def process_nodes(node):
    if not node or "children" not in node:
        return
    
    # Process all children recursively
    for child in node["children"]:
        process_nodes(child)
    
    # Convert P followed by INPUT into LABEL
    children = node.get("children", [])
    for i in range(len(children) - 1):
        if (children[i].get("tag") == "P" and 
            children[i+1].get("tag") == "INPUT"):
            children[i]["tag"] = "LABEL"
    
    # Identify and set NAVBAR (a full-width div at the top)
    for child in children:
        if (child.get("tag") == "DIV" and 
            child.get("node", {}).get("x") == 0.0 and 
            child.get("node", {}).get("y") == 0.0 and 
            child.get("node", {}).get("width", 0) >= 600): # Assuming body_width is around 668
            child["tag"] = "NAVBAR"
    
    # Convert Group DIV with at least 2 LI elements into UL
    for child in children:
        if (child.get("tag") == "DIV" and 
            count_list_items(child) >= 2):  # Require at least 2 LI elements
            child["tag"] = "UL"
    
    # Convert Group DIV containing inputs/buttons into FORM
    for child in children:
        if child.get("tag") == "DIV":
            form_elements = count_form_elements(child)
            if form_elements >= 2:
                child["tag"] = "FORM"

def count_form_elements(node):
    """Count INPUT and BUTTON elements in direct and nested children."""
    count = 0
    if not node or "children" not in node:
        return count
    
    # Check direct children
    for child in node.get("children", []):
        if child.get("tag") == "FORM":
            return 0  # Skip if any direct child is already a FORM
        if child.get("tag") in ["INPUT", "BUTTON"]:
            count += 1
    
    # Check nested children
    for child in node.get("children", []):
        count += count_form_elements(child)
    
    return count

def count_list_items(node):
    """Count LI elements in direct and first-level indirect children, avoiding deeper UL nesting."""
    if not node or "children" not in node:
        return 0

    count = 0
    for child in node.get("children", []):
        if child.get("tag") == "LI":
            count += 1
        elif child.get("tag") != "UL":  # Stop at existing ULs
            count += sum(1 for grandchild in child.get("children", []) if grandchild.get("tag") == "LI")

    return count

def generate_random_color():
    """Generate a random color with some transparency"""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgba({r},{g},{b},0.3)"

def draw_tags_on_svg_file(data, svg_input_file, svg_output_file=None):
    """
    Draw bounding boxes and tags on a copy of an existing SVG file.

    Args:
        data (dict): The data containing node information.
        svg_input_file (str): Path to the original SVG file.
        svg_output_file (str, optional): Path to save the modified SVG 
                                          (if None, will use input_file + "_tagged.svg").
    """
    
    # Create output filename if not provided
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"
    
    # Make a copy of the original SVG file
    shutil.copy2(svg_input_file, svg_output_file)
    
    # Parse the original SVG to modify it directly
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()
    
    # Get SVG dimensions
    frame_width = root.get('width', str(data.get("node", {}).get("width", "1000"))).replace('px', '')
    frame_height = root.get('height', str(data.get("node", {}).get("height", "1000"))).replace('px', '')

    # Add style element
    style_element = etree.SubElement(root, 'style')
    style_element.text = """
        .tag-box { stroke: #000000; stroke-width: 1; fill-opacity: 0.3; }
        .tag-text { font-family: Arial; font-size: 10px; }
        .tag-label { fill: white; stroke: #000000; stroke-width: 0.5; rx: 3; ry: 3; }
    """
    
    # Create a group element for our tags
    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    
    # Track assigned colors by tag type
    tag_colors = {}

    def draw_element(element, parent_element):
        """Recursively draw bounding boxes and labels for elements."""
        if not element or "node" not in element:
            return
            
        tag = element.get("tag", "UNKNOWN")
        
        # Assign consistent colors to tag types
        if tag not in tag_colors:
            tag_colors[tag] = generate_random_color()
        color = tag_colors[tag]
        
        # Get node position and dimensions
        node = element["node"]
        x, y = node.get("x", 0), node.get("y", 0)
        width, height = node.get("width", 50), node.get("height", 50)
        
        # Create a group for this element
        group = etree.SubElement(parent_element, 'g')

        # Draw rectangle for bounding box
        rect = etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "class": "tag-box",
            "fill": color,
            "stroke": "black",
            "stroke-width": "1"
        })

        # Label Background
        label_width = max(80, len(tag) * 7)
        label_height = 40
        
        label_bg = etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(label_width),
            "height": str(label_height),
            "rx": "3",
            "ry": "3",
            "fill": "white",
            "fill-opacity": "0.7",
            "stroke": "black",
            "stroke-width": "0.5"
        })
        
        # Add text labels
        etree.SubElement(group, 'text', {
            "x": str(x + 3),
            "y": str(y + 12),
            "class": "tag-text",
            "fill": "black"
        }).text = tag
        
        etree.SubElement(group, 'text', {
            "x": str(x + 3),
            "y": str(y + 24),
            "class": "tag-text",
            "fill": "black"
        }).text = f"x:{x:.1f}, y:{y:.1f}"
        
        etree.SubElement(group, 'text', {
            "x": str(x + 3),
            "y": str(y + 36),
            "class": "tag-text",
            "fill": "black"
        }).text = f"w:{width:.1f}, h:{height:.1f}"
        
        # Process children recursively
        for child in element.get("children", []):
            draw_element(child, group)

    # Start drawing from the root
    draw_element(data, tag_group)
    
    # Save the modified SVG file
    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"SVG visualization created at {svg_output_file}")
    
    # Open the SVG file in the default viewer
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")

def process_figma_json(input_file, output_file, svg_file=None):
    """
    Process a Figma JSON file, predicting tags for UNK nodes.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        svg_file: Path to the original SVG file (optional)
    """
    # Load model and preprocessing tools
    model, label_encoder, ohe, imputer, scaler = load_model_and_encoders()
    
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if os.path.exists('features_with_prediction.csv'):
        os.remove('features_with_prediction.csv')
        
    # Predict tags recursively
    predict_tag(data, 0, None, None, None, None, model, label_encoder, ohe, imputer, scaler)
    
    # Post processing
    data = post_process_tags(data)
    
    # Check if SVG file is provided
    if svg_file:
        # Generate the output SVG file path
        svg_output = os.path.splitext(svg_file)[0] + "_tagged.svg"
        
        # Draw tags on a copy of the SVG
        draw_tags_on_svg_file(data, svg_file, svg_output)
    
    # Save processed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {input_file}. Output saved to {output_file}")
    
# Example usage
if __name__ == "__main__":
    process_figma_json("input.json", "output.json", "input.svg")