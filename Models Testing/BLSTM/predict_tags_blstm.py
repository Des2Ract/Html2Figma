import json
import torch
import numpy as np
import os
import h5py
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from lxml import etree
import webbrowser
import shutil
import random
from typing import Dict, List, Optional
from tqdm import tqdm
import sys
import math

utils_path = os.path.abspath(os.path.join(os.getcwd(), "../../Utils/"))
sys.path.append(utils_path)

from utils import verb_ratio, is_near_gray, color_difference, find_nearest_text_node, collect_text_nodes, count_all_descendants, count_chars_to_end, get_center_of_weight

# Disabling oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Global variables used in normalization
body_width = None
body_height = None
num_nodes = None
num_chars = None

class OptimizedFigmaBLSTM(nn.Module):
    """Optimizing Bidirectional LSTM model for HTML tag prediction"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(OptimizedFigmaBLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.size()
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        output = self.dropout(output)
        
        if batch_size > 1:
            output_reshaped = output.reshape(-1, output.size(-1))
            output_reshaped = self.batch_norm(output_reshaped)
            output = output_reshaped.reshape(batch_size, seq_len, -1)
        else:
            output_reshaped = output.view(-1, output.size(-1))
            output_normed = self.batch_norm(output_reshaped)
            output = output_normed.view(batch_size, seq_len, -1)
        
        logits = self.fc(output)
        return logits

class FigmaHTMLFeatureExtractor:
    def __init__(
        self,
        semantic_model_name: str = 'all-MiniLM-L6-v2',
        node_type_embedding_dim: int = 50
    ):
        """Initializing feature extractor with SentenceTransformer and node type embeddings"""
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.text_embedding_dim = 384
        self.node_name_embedding_dim = 384
        self.node_types = self._get_node_types()
        self.node_type_to_idx = {node_type: idx for idx, node_type in enumerate(self.node_types)}
        self.node_type_embedding_dim = node_type_embedding_dim
        self.node_type_embedding_layer = nn.Embedding(len(self.node_types), self.node_type_embedding_dim)
        self.tag_mapping = self._get_tag_mapping()
        self.custom_tag_removal_pattern = self._get_custom_tag_removal_pattern()
        self.default_tag = "DIV"
        self.icon_like_node_types = {"VECTOR", "INSTANCE", "COMPONENT", "SHAPE", "SVG_ICON"}
        self.stats = {
            "nodes_processed": 0,
            "tag_mappings": {},
            "unique_node_types": set()
        }

    def _get_node_types(self) -> List[str]:
        """Defining most of Figma node types"""
        return [
            "TEXT", "RECTANGLE", "GROUP", "ELLIPSE", "FRAME", "VECTOR", "STAR", "LINE",
            "POLYGON", "BOOLEAN_OPERATION", "SLICE", "COMPONENT", "INSTANCE", "COMPONENT_SET",
            "DOCUMENT", "CANVAS", "SECTION", "SHAPE_WITH_TEXT", "STICKY", "TABLE", "WASHI_TAPE",
            "CONNECTOR", "HIGHLIGHT", "WIDGET", "EMBED", "LINK", "LINK_UNFURL", "MEDIA", "CODE_BLOCK",
            "STAMP", "COMMENT", "FREEFORM", "TIMELINE", "STICKER", "SHAPE", "ARROW", "CALL_OUT",
            "FLOW", "TEXT_AREA", "TEXT_FIELD", "BUTTON", "CHECKBOX", "RADIO", "TOGGLE", "SLIDER",
            "DROPDOWN", "COMBOBOX", "LIST", "TABLE_CELL", "TABLE_ROW", "TABLE_COLUMN", "TABLE_SECTION",
            "TABLE_HEADER", "TABLE_FOOTER", "TABLE_BODY", "TABLE_CAPTION", "TABLE_COLGROUP", "TABLE_COL",
            "TABLE_THEAD", "TABLE_TBODY", "TABLE_TFOOT", "TABLE_TR", "TABLE_TH", "TABLE_TD",
            "UNKNOWN_TYPE"
        ]

    def _get_tag_mapping(self) -> Dict[str, str]:
        """Clustring raw HTML tags to chosen ones"""
        return {
            "ARTICLE": "DIV", "DIV": "DIV", "FIGURE": "DIV", "FOOTER": "DIV",
            "HEADER": "DIV", "NAV": "DIV", "MAIN": "DIV", "IFRAME": "DIV",
            "BODY": "DIV", "FORM": "DIV", "TABLE": "DIV", "THEAD": "DIV",
            "TBODY": "DIV", "SECTION": "DIV", "ASIDE": "DIV",
            "UL": "LIST", "OL": "LIST", "DL": "LIST",
            "H1": "P", "H2": "P", "H3": "P", "H4": "P", "H5": "P", "H6": "P",
            "SUP": "P", "SUB": "P", "BIG": "P", "P": "P", "CAPTION": "P", "FIGCAPTION": "P",
            "B": "P", "EM": "P", "I": "P", "TD": "P", "TH": "P", "TR": "P", "PRE": "P",
            "U": "P", "TIME": "P", "TXT": "P", "ABBR": "P", "SMALL": "P", "STRONG": "P",
            "SUMMARY": "P", "SPAN": "P", "LABEL": "P", "LI": "P", "DD": "P",
            "A": "P", "BLOCKQUOTE": "P", "CODE": "P",
            "PICTURE": "IMG", "VIDEO": "IMG",
            "SELECT": "INPUT", "TEXTAREA": "INPUT",
            "VECTOR": "SVG", "ICON": "SVG",
            "UNK": "CONTAINER"
        }

    def _get_custom_tag_removal_pattern(self) -> str:
        """Defining regex pattern for invalid tag removal"""
        return r'[-:]|\b(DETAILS|CANVAS|FIELDSET|COLGROUP|COL|CNX|ADDRESS|CITE|S|DEL|LEGEND|BDI|LOGO|OBJECT|OPTGROUP|CENTER|FRONT|Q|SEARCH|SLOT|AD|ADSLOT|BLINK|BOLD|COMMENTS|DATA|DIALOG|EMBED|EMPHASIS|FONT|H7|HGROUP|INS|INTERACTION|ITALIC|ITEMTEMPLATE|MATH|MENU|MI|MN|MO|MROW|MSUP|NOBR|OFFER|PATH|PROGRESS|STRIKE|SWAL|TEXT|TITLE|TT|VAR|VEV|W|WBR|COUNTRY|ESI:INCLUDE|HTTPS:|LOGIN|NOCSRIPT|PERSONAL|STONG|CONTENT|DELIVERY|LEFT|MSUBSUP|KBD|ROOT|PARAGRAPH|BE|AI2SVELTEWRAP|BANNER|PHOTO1)\b'

    def clean_and_map_tag(self, raw_tag: str) -> str:
        """Cleaning and mapping raw HTML tag to simpler tag group"""
        import re
        if not raw_tag:
            return self.default_tag

        raw_tag = raw_tag.upper()
        cleaned = self.tag_mapping.get(raw_tag, raw_tag)
        if re.search(self.custom_tag_removal_pattern, cleaned, re.IGNORECASE):
            cleaned = self.default_tag

        final = self.tag_mapping.get(cleaned, cleaned)
        if final != raw_tag:
            self.stats["tag_mappings"][raw_tag] = self.stats["tag_mappings"].get(raw_tag, 0) + 1

        return final

    def determine_bioes_label(self, base_tag: str) -> str:
        """Assigning BIOES label for a tag"""
        if base_tag == "CONTAINER":
            return "B_CONTAINER"
        return base_tag

    def extract_features(
        self, 
        node_data_item: Dict, 
        current_body_width: float,
        sequence_id: str,
        parent_node_height: Optional[float] = None,
        parent_base_tag: Optional[str] = None,
        depth: int = 0,
        position_in_siblings: int = 0,
        total_siblings: int = 1,
        text_nodes: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Extracting features recursively from a node and its children"""
        global body_width, body_height, num_nodes, num_chars

        features_and_labels_list = []
        node_dict = node_data_item.get("node", {})
        raw_tag = node_data_item.get("tag", "UNK").upper()

        if text_nodes is None:
            text_nodes = collect_text_nodes(node_data_item)

        # Updating global variables
        if not body_width or body_width == 0:
            body_width = float(node_dict.get("width", 1000.0))
        if not body_height or body_height == 0:
            body_height = float(node_dict.get("height", 1000.0))
        num_children_to_end = count_all_descendants(node_data_item)
        if not num_nodes or num_nodes == 0:
            num_nodes = num_children_to_end
        chars_count_to_end = count_chars_to_end(node_data_item)
        if not num_chars or num_chars == 0:
            num_chars = chars_count_to_end

        # Determining base tag and BIOES label
        has_children = bool(node_data_item.get("children"))
        base_tag = self.clean_and_map_tag(raw_tag)
        bioes_label = self.determine_bioes_label(base_tag)

        # Extracting node-type embedding
        node_type_str = node_dict.get("type", "UNKNOWN_TYPE")
        self.stats["unique_node_types"].add(node_type_str)
        node_type_idx = self.node_type_to_idx.get(node_type_str, self.node_type_to_idx["UNKNOWN_TYPE"])
        node_type_emb = (
            self.node_type_embedding_layer(torch.tensor(node_type_idx, dtype=torch.long))
            .detach()
            .cpu()
            .numpy()
        )

        # Extracting text embedding
        text_content = node_dict.get("characters", "").strip()
        if node_type_str == "TEXT" and text_content:
            text_emb = self.semantic_model.encode(text_content)
            is_verb = verb_ratio(text_content)
        else:
            text_emb = np.zeros(self.text_embedding_dim)
            is_verb = 0.0

        # Extracting node name embedding
        node_name = node_data_item.get("name", "").strip()
        if node_name and (node_type_str in self.icon_like_node_types or "icon" in node_name.lower()):
            node_name_emb = self.semantic_model.encode(node_name)
        else:
            node_name_emb = np.zeros(self.node_name_embedding_dim)

        # Extracting numerical and structural features
        eps = 1e-6
        node_width = float(node_dict.get("width", 0))
        node_height = float(node_dict.get("height", 0))
        aspect_ratio = node_width / (node_height + eps) if node_height > 0 else 0.0
        normalized_width = node_width / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_height = node_height / (parent_node_height + eps) if parent_node_height and parent_node_height > 0 else 0.0

        x_position = float(node_dict.get("x", 0))
        y_position = float(node_dict.get("y", 0))
        normalized_x = x_position / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_y = y_position / (body_height + eps) if body_height > 0 else 0.0

        normalized_depth = min(depth / 20.0, 1.0)
        normalized_position = position_in_siblings / (total_siblings + eps)

        # Extracting background color and placeholder status
        has_placeholder = 0.0
        bg_color = [0.0, 0.0, 0.0, 0.0]
        fills = node_dict.get("fills", [])
        fills = [fill for fill in fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
        if fills and isinstance(fills, list) and len(fills) > 0:
            color = fills[0].get("color", {})
            bg_color = [
                float(color.get("r", 0.0)),
                float(color.get("g", 0.0)),
                float(color.get("b", 0.0)),
                float(color.get("a", 0.0)),
            ]
            r, g, b = (
                int(color.get("r", 0) * 255),
                int(color.get("g", 0) * 255),
                int(color.get("b", 0) * 255),
            )
            if is_near_gray(r, g, b):
                has_placeholder = 1.0

        font_size = float(node_dict.get("fontSize", 0.0)) / 100.0
        flex_direction = 1.0 if node_dict.get("flexDirection", "") == "column" else 0.0

        # Extracting nearest text node distance and center of weight
        nearest_text_distance = find_nearest_text_node(node_data_item, text_nodes)
        normalized_text_dist = (nearest_text_distance + 0.01) / (math.sqrt((node_width + 0.001) * (node_height + 0.001)) if math.sqrt((node_width + 0.001) * (node_height + 0.001)) else 1)
        center_of_weight_diff = get_center_of_weight(node_data_item)

        # Combining features into one vector
        feature_vector = np.concatenate([
            node_type_emb,
            text_emb,
            node_name_emb,
            np.array([
                normalized_width, normalized_height, aspect_ratio,
                normalized_x, normalized_y,
                normalized_depth, normalized_position,
                *bg_color, font_size, flex_direction,
                is_verb, has_placeholder, normalized_text_dist, center_of_weight_diff
            ], dtype=np.float32)
        ])

        features_and_labels_list.append({
            "feature_vector": feature_vector.astype(np.float32),
            "tag": bioes_label,
            "base_tag": base_tag,
            "node_data": node_data_item
        })

        self.stats["nodes_processed"] += 1

        # Recursing into children
        if has_children:
            children = node_data_item["children"]
            total_children = len(children)
            for idx, child in enumerate(children):
                child_features = self.extract_features(
                    node_data_item=child,
                    current_body_width=current_body_width,
                    sequence_id=sequence_id,
                    parent_node_height=node_height,
                    parent_base_tag=base_tag,
                    depth=depth + 1,
                    position_in_siblings=idx,
                    total_siblings=total_children,
                    text_nodes=text_nodes
                )
                features_and_labels_list.extend(child_features)

            if bioes_label == "B_CONTAINER":
                zero_vector = np.zeros_like(feature_vector, dtype=np.float32)
                features_and_labels_list.append({
                    "feature_vector": zero_vector,
                    "tag": "E_CONTAINER",
                    "base_tag": "E_CONTAINER",
                    "node_data": None
                })

        if depth == 0:
            one_vector = np.ones_like(feature_vector, dtype=np.float32)
            features_and_labels_list.append({
                "feature_vector": one_vector,
                "tag": "E_WEBSITE",
                "base_tag": "E_WEBSITE",
                "node_data": None
            })

        return features_and_labels_list

def load_model_and_metadata(model_path: str, device: str = None):
    """Loading BLSTM model and metadata from checkpoint"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['model_config']
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']

    model = OptimizedFigmaBLSTM(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        output_dim=num_classes,
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, label_encoder, input_dim, model_config

def generate_random_color():
    """Generating random RGBA color for SVG visualization."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgba({r},{g},{b},0.3)"

def draw_tags_on_svg_file(
    data: Dict,
    features: List[Dict],
    predicted_tags: List[str],
    svg_input_file: str,
    svg_output_file: Optional[str] = None
):
    """
    Draw bounding boxes and tags on a copy of an existing SVG file

    Args:
        data : The data containing node information.
        svg_input_file : Path to the original SVG file.
        svg_output_file (optional): Path to save the modified SVG (if None, will use input_file + "_tagged.svg")
    """
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"

    shutil.copy2(svg_input_file, svg_output_file)

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()

    def compute_max_bounds(element, max_x=0, max_y=0):
        """Compute maximum bounds for SVG viewport"""
        if "node" in element:
            node = element["node"]
            x = float(node.get("x", 0))
            y = float(node.get("y", 0))
            width = float(node.get("width", 0))
            height = float(node.get("height", 0))
            max_x = max(max_x, x + width + 100)
            max_y = max(max_y, y + height + 100)
        for child in element.get("children", []):
            max_x, max_y = compute_max_bounds(child, max_x, max_y)
        return max_x, max_y

    max_x, max_y = compute_max_bounds(data)

    # Set SVG bounds and remove overflow styles
    root.attrib["width"] = str(int(max_x))
    root.attrib["height"] = str(int(max_y))
    root.attrib["viewBox"] = f"0 0 {int(max_x)} {int(max_y)}"
    root.attrib.pop("style", None)

    # Remove nested <svg> elements
    for nested_svg in root.findall(".//{http://www.w3.org/2000/svg}svg"):
        parent = nested_svg.getparent()
        if parent is not None:
            parent.remove(nested_svg)

    # Add CSS style block
    style_el = etree.SubElement(root, 'style')
    style_el.text = """
        .tag-box {
            stroke: #000;
            stroke-width: 1;
            fill-opacity: 0.15;
            filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.4));
        }
        .changed-tag {
            fill: #ff0000;
            fill-opacity: 0.25;
            stroke: #ff0000;
            stroke-width: 2;
            filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.4));
        }
        .tag-text {
            font-family: Arial;
            font-size: 12px;
            font-weight: bold;
        }
    """

    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    tag_colors = {}
    drawn_positions = set()

    def draw_element(feat_dict, tag, parent_element):
        """Draw a single element with its tag and bounding box"""
        node_data_item = feat_dict.get("node_data")
        if not node_data_item or not node_data_item.get("node"):
            return

        base_tag = feat_dict.get("base_tag", tag)
        node = node_data_item["node"]
        x, y = float(node.get("x", 0.0)), float(node.get("y", 0.0))
        width, height = float(node.get("width", 50.0)), float(node.get("height", 50.0))

        color = tag_colors.setdefault(tag, generate_random_color())
        group = etree.SubElement(parent_element, 'g')
        is_changed = tag != base_tag
        rect_class = "changed-tag" if is_changed else "tag-box"

        etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": "5",
            "ry": "5",
            "class": rect_class,
            "fill": color,
        })

        lines = [f"{base_tag} → {tag}" if is_changed else tag, f"x:{x:.0f}, y:{y:.0f}"]
        label_padding = 4
        label_line_height = 14
        label_width = max(80, len(lines[0]) * 7)
        label_height = label_line_height * len(lines) + label_padding

        ideal_label_y = y + 2
        label_x = x + 4
        offset_step = 12
        max_attempts = 10
        for i in range(max_attempts):
            label_y = ideal_label_y - i * offset_step
            key = (round(label_x), round(label_y))
            if key not in drawn_positions:
                drawn_positions.add(key)
                break

        label_y = max(0, label_y)
        label_x = min(label_x, max_x - label_width - 2)

        etree.SubElement(group, 'rect', {
            "x": str(label_x),
            "y": str(label_y),
            "width": str(label_width),
            "height": str(label_height),
            "rx": "3",
            "ry": "3",
            "fill": "#ffffff",
            "fill-opacity": "0.85",
            "stroke": "#000000",
            "stroke-width": "0.5"
        })

        for i, line in enumerate(lines):
            etree.SubElement(group, 'text', {
                "x": str(label_x + 5),
                "y": str(label_y + (i + 1) * label_line_height - 4),
                "class": "tag-text",
                "fill": "black"
            }).text = line

    def is_priority(element):
        """Determine if an element should be drawn in the foreground"""
        tag = element.get("tag", "").lower()
        if tag == "p":
            return None
        return tag in {"button", "input", "card", "list", "navbar", "footer", "checkbox", "li"}

    # Recursively draw elements by priority
    def draw_by_priority(element, parent_element, priority: bool):
        """Draw elements by priority to manage layering."""
        if not element or "node" not in element:
            return
        priority_status = is_priority(element)
        if priority_status is None:
            return  # Skip this element and its drawing, but not children
        if priority_status == priority:
            feat_dict = next((f for f in features if f.get("node_data") == element), None)
            if feat_dict:
                idx = features.index(feat_dict)
                draw_element(feat_dict, predicted_tags[idx], parent_element)
        for child in element.get("children", []):
            draw_by_priority(child, parent_element, priority)

    # Draw non-priority (background)
    draw_by_priority(data, tag_group, priority=False)

    # Draw priority (foreground)
    draw_by_priority(data, tag_group, priority=True)

    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"✅ SVG visualization saved to: {svg_output_file}")
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")

def post_process_tags(data: Dict, features: List[Dict], predicted_tags: List[str]):
    """Post-processing predicted tags"""
    global body_width

    def process_nodes(node_item: Dict, feature_index: int) -> int:
        """Recursively assigning predicted tags and applying rule based"""
        if not node_item or "children" not in node_item:
            return feature_index

        node_dict = node_item.get("node", {})
        node_type = node_dict.get("type", "")
        base_tag = features[feature_index].get("base_tag", predicted_tags[feature_index])

        if node_type == "GROUP":
            node_item["tag"] = "DIV"
            node_item["base_tag"] = base_tag
            predicted_tags[feature_index] = "DIV"
        elif node_type == "TEXT":
            node_item["tag"] = "P"
            node_item["base_tag"] = base_tag
            predicted_tags[feature_index] = "P"
        elif node_type in ["SVG", "VECTOR"] or node_item.get("name", "").startswith("ICON"):
            node_item["tag"] = "SVG"
            node_item["base_tag"] = base_tag
            predicted_tags[feature_index] = "SVG"
        elif node_type == "LINE":
            node_item["tag"] = "HR"
            node_item["base_tag"] = base_tag
            predicted_tags[feature_index] = "HR"
        elif (fills := node_dict.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills):
            node_item["tag"] = "IMG"
            node_item["base_tag"] = base_tag
            predicted_tags[feature_index] = "IMG"
        else:
            node_item["tag"] = predicted_tags[feature_index]
            node_item["base_tag"] = base_tag

        feature_index += 1

        for child in node_item.get("children", []):
            feature_index = process_nodes(child, feature_index)

        return feature_index

    def apply_post_processing_rules(node_item: Dict, body_width: float):
        """Applying rule based for grouping"""
        if not node_item or "children" not in node_item:
            return

        children = node_item["children"]
        for i in range(len(children) - 1):
            if children[i].get("tag") == "P" and children[i + 1].get("tag") == "INPUT":
                children[i]["tag"] = "LABEL"
                children[i]["base_tag"] = children[i].get("base_tag", "P")

        for child in children:
            ch_node = child.get("node", {})
            if (
                child.get("tag") == "DIV"
                and float(ch_node.get("x", 0.0)) == 0.0
                and float(ch_node.get("y", 0.0)) == 0.0
                and abs(float(ch_node.get("width", 0.0)) - body_width) < 5
                and float(ch_node.get("height", 0.0)) < body_width / 10
            ):
                child["tag"] = "NAVBAR"
                child["base_tag"] = child.get("base_tag", "DIV")

        for child in children:
            if child.get("tag") == "DIV":
                li_count = sum(1 for c in child.get("children", []) if c.get("tag") == "LI")
                if li_count >= 2:
                    child["tag"] = "LIST"
                    child["base_tag"] = child.get("base_tag", "DIV")

        for child in children:
            if child.get("tag") == "DIV":
                form_elems = sum(1 for c in child.get("children", []) if c.get("tag") in ["INPUT", "BUTTON"])
                if form_elems >= 2:
                    child["tag"] = "FORM"
                    child["base_tag"] = child.get("base_tag", "DIV")

        for child in children:
            apply_post_processing_rules(child, body_width)

    process_nodes(data, 0)
    body_width = float(data.get("node", {}).get("width", 1000.0)) or 1000.0
    apply_post_processing_rules(data, body_width)

def predict_tags_blstm(
    input_file: str,
    output_file: str,
    model_path: str,
    svg_file: Optional[str] = None
):
    """Processing Figma JSON to predict and visualize HTML tags"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, label_encoder, input_dim, model_config = load_model_and_metadata(model_path, device)
    print("Model and metadata loaded successfully.")

    extractor = FigmaHTMLFeatureExtractor(
        semantic_model_name='all-MiniLM-L6-v2',
        node_type_embedding_dim=50
    )

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sequence_id = os.path.basename(input_file).replace(".json", "")
    body_width = float(data.get("node", {}).get("width", 1000.0)) or 1000.0

    print("Extracting features...")
    features = extractor.extract_features(
        node_data_item=data,
        current_body_width=body_width,
        sequence_id=sequence_id
    )

    feature_vectors = np.stack([f["feature_vector"] for f in features]).astype(np.float32)
    seq_len = feature_vectors.shape[0]
    padded_features = torch.from_numpy(feature_vectors).unsqueeze(0).to(device)
    lengths = torch.tensor([seq_len], dtype=torch.long)

    print("Predicting tags...")
    with torch.no_grad():
        outputs = model(padded_features, lengths)
        predicted_indices = torch.argmax(outputs, dim=2).cpu().numpy().flatten()
        predicted_tags = label_encoder.inverse_transform(predicted_indices)

    print("Post-processing tags...")
    post_process_tags(data, features, list(predicted_tags))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Processed JSON saved to: {output_file}")

    if svg_file:
        print("Drawing annotations on SVG...")
        draw_tags_on_svg_file(data, features, list(predicted_tags), svg_file)

    print("\nProcessing Statistics:")
    print(f"  Nodes processed: {extractor.stats['nodes_processed']}")
    print(f"  Unique node types seen: {len(extractor.stats['unique_node_types'])}")
    print(f"  Total tag mappings applied: {sum(extractor.stats['tag_mappings'].values())}")

if __name__ == "__main__":
    predict_tags_blstm(
        input_file="../Data/input.json",
        output_file="../Data/output.json",
        model_path="../Models/figma_blstm_final_model.pt",
        svg_file="../Data/input.svg"
    )