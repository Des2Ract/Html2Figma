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

# Placeholder TreeLSTMCell (to be replaced with actual implementation)
class TreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        # Simplified placeholder; actual implementation should handle tree-specific gates
        self.fc = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)  # Gates: i, f, o, u

    def forward(self, x, h_children, c_children):
        # Placeholder logic; replace with proper TreeLSTM equations
        combined = x
        if h_children:
            combined = torch.cat([x, sum(h_children) / len(h_children)], dim=-1)
        gates = self.fc(combined)
        h = torch.tanh(gates)  # Simplified
        c = h  # Simplified
        return h, c

class TreeBLSTM(nn.Module):
    """Bidirectional Tree LSTM model for hierarchical processing."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TreeBLSTM, self).__init__()
        self.forward_cell = TreeLSTMCell(input_dim, hidden_dim)
        self.backward_cell = TreeLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def _forward_backward(self, node, logits_list):
        if not node.children:
            x = node.feature_vector.to(device)
            h_f, c_f = self.forward_cell(x, [], [])
            h_b, c_b = self.backward_cell(x, [], [])
        else:
            h_children_f = []
            c_children_f = []
            h_children_b = []
            c_children_b = []
            for child in node.children:
                h_f_child, c_f_child, h_b_child, c_b_child = self._forward_backward(child, logits_list)
                h_children_f.append(h_f_child)
                c_children_f.append(c_f_child)
                h_children_b.append(h_b_child)
                c_children_b.append(c_b_child)
            x = node.feature_vector.to(device)
            h_f, c_f = self.forward_cell(x, h_children_f, c_children_f)
            h_b, c_b = self.backward_cell(x, h_children_b, c_children_b)
        h_combined = torch.cat([h_f, h_b], dim=-1)
        h_combined = self.dropout(h_combined)
        logits = self.fc(h_combined)
        logits_list.append(logits)
        return h_f, c_f, h_b, c_b

    def predict_all(self, tree):
        logits_list = []
        self._forward_backward(tree, logits_list)
        return logits_list

class TreeNode:
    def __init__(self, feature_vector, node_data):
        self.feature_vector = torch.from_numpy(feature_vector).float()
        self.node_data = node_data
        self.children = []
        self.predicted_tag = None

    def get_nodes_depth_first(self):
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_nodes_depth_first())
        return nodes

class FigmaHTMLFeatureExtractor:
    def __init__(
        self,
        semantic_model_name: str = 'all-MiniLM-L6-v2',
        node_type_embedding_dim: int = 50
    ):
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.text_embedding_dim = 384
        self.node_name_embedding_dim = 384
        self.node_types = self._get_node_types()
        self.node_type_to_idx = {nt: idx for idx, nt in enumerate(self.node_types)}
        self.node_type_embedding_dim = node_type_embedding_dim
        self.node_type_embedding_layer = nn.Embedding(len(self.node_types), node_type_embedding_dim)
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
        return r'[-:]|\b(DETAILS|CANVAS|FIELDSET|COLGROUP|COL|CNX|ADDRESS|CITE|S|DEL|LEGEND|BDI|LOGO|OBJECT|OPTGROUP|CENTER|FRONT|Q|SEARCH|SLOT|AD|ADSLOT|BLINK|BOLD|COMMENTS|DATA|DIALOG|EMBED|EMPHASIS|FONT|H7|HGROUP|INS|INTERACTION|ITALIC|ITEMTEMPLATE|MATH|MENU|MI|MN|MO|MROW|MSUP|NOBR|OFFER|PATH|PROGRESS|STRIKE|SWAL|TEXT|TITLE|TT|VAR|VEV|W|WBR|COUNTRY|ESI:INCLUDE|HTTPS:|LOGIN|NOCSRIPT|PERSONAL|STONG|CONTENT|DELIVERY|LEFT|MSUBSUP|KBD|ROOT|PARAGRAPH|BE|AI2SVELTEWRAP|BANNER|PHOTO1)\b'

    def clean_and_map_tag(self, raw_tag: str) -> str:
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

    def extract_features(
        self,
        node_data_item: Dict,
        current_body_width: float,
        sequence_id: str,
        parent_node_height: Optional[float] = None,
        depth: int = 0,
        position_in_siblings: int = 0,
        total_siblings: int = 1
    ) -> TreeNode:
        node_dict = node_data_item.get("node", {})
        node_type_str = node_dict.get("type", "UNKNOWN_TYPE")
        self.stats["unique_node_types"].add(node_type_str)
        node_type_idx = self.node_type_to_idx.get(node_type_str, self.node_type_to_idx["UNKNOWN_TYPE"])
        node_type_emb = self.node_type_embedding_layer(torch.tensor(node_type_idx)).detach().cpu().numpy()

        text_content = node_dict.get("characters", "").strip()
        text_emb = self.semantic_model.encode(text_content) if node_type_str == "TEXT" and text_content else np.zeros(self.text_embedding_dim)

        node_name = node_data_item.get("name", "").strip()
        node_name_emb = self.semantic_model.encode(node_name) if node_name and (node_type_str in self.icon_like_node_types or "icon" in node_name.lower()) else np.zeros(self.node_name_embedding_dim)

        eps = 1e-6
        node_width = float(node_dict.get("width", 0))
        node_height = float(node_dict.get("height", 0))
        aspect_ratio = node_width / (node_height + eps) if node_height > 0 else 0.0
        normalized_width = node_width / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_height = node_height / (parent_node_height + eps) if parent_node_height else 0.0
        x_position = float(node_dict.get("x", 0))
        y_position = float(node_dict.get("y", 0))
        normalized_x = x_position / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_y = y_position / (parent_node_height + eps) if parent_node_height else 0.0
        normalized_depth = min(depth / 20.0, 1.0)
        normalized_position = position_in_siblings / (total_siblings + eps)
        bg_color = [0.0, 0.0, 0.0, 0.0]
        fills = node_dict.get("fills", [])
        if fills and isinstance(fills, list) and len(fills) > 0:
            color = fills[0].get("color", {})
            bg_color = [float(color.get(k, 0.0)) for k in ("r", "g", "b", "a")]
        font_size = float(node_dict.get("fontSize", 0.0)) / 100.0
        flex_direction = 1.0 if node_dict.get("flexDirection", "") == "column" else 0.0

        feature_vector = np.concatenate([
            node_type_emb,
            text_emb,
            node_name_emb,
            np.array([normalized_width, normalized_height, aspect_ratio, normalized_x, normalized_y, normalized_depth, normalized_position, *bg_color, font_size, flex_direction], dtype=np.float32)
        ])

        tree_node = TreeNode(feature_vector, node_data_item)
        self.stats["nodes_processed"] += 1

        children = node_data_item.get("children", [])
        total_children = len(children)
        for idx, child in enumerate(children):
            child_node = self.extract_features(
                child,
                current_body_width,
                sequence_id,
                node_height,
                depth + 1,
                idx,
                total_children
            )
            tree_node.children.append(child_node)

        return tree_node

def load_model_and_metadata(model_path: str, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']

    model = TreeBLSTM(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        output_dim=num_classes
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, label_encoder, input_dim, model_config

def generate_random_color():
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
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"
    shutil.copy2(svg_input_file, svg_output_file)
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()
    style_el = etree.SubElement(root, 'style')
    style_el.text = """
        .tag-box { stroke: #000000; stroke-width: 1; fill-opacity: 0.3; }
        .tag-text { font-family: Arial; font-size: 10px; }
        .tag-label { fill: white; stroke: #000000; stroke-width: 0.5; rx: 3; ry: 3; }
    """
    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    tag_colors = {}

    for feat_dict, tag in zip(features, predicted_tags):
        node_data_item = feat_dict.get("node_data")
        if not node_data_item or not node_data_item.get("node"):
            continue
        node = node_data_item["node"]
        x, y = float(node.get("x", 0.0)), float(node.get("y", 0.0))
        width, height = float(node.get("width", 50.0)), float(node.get("height", 50.0))
        if tag not in tag_colors:
            tag_colors[tag] = generate_random_color()
        color = tag_colors[tag]
        grp = etree.SubElement(tag_group, 'g')
        etree.SubElement(grp, 'rect', {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "class": "tag-box", "fill": color, "stroke": "black", "stroke-width": "1"
        })
        label_w = max(80, len(tag) * 7)
        label_h = 40
        etree.SubElement(grp, 'rect', {
            "x": str(x), "y": str(y), "width": str(label_w), "height": str(label_h),
            "rx": "3", "ry": "3", "fill": "white", "fill-opacity": "0.7", "stroke": "black", "stroke-width": "0.5"
        })
        etree.SubElement(grp, 'text', {
            "x": str(x + 3), "y": str(y + 12), "class": "tag-text", "fill": "black"
        }).text = tag
        etree.SubElement(grp, 'text', {
            "x": str(x + 3), "y": str(y + 24), "class": "tag-text", "fill": "black"
        }).text = f"x:{x:.1f}, y:{y:.1f}"
        etree.SubElement(grp, 'text', {
            "x": str(x + 3), "y": str(y + 36), "class": "tag-text", "fill": "black"
        }).text = f"w:{width:.1f}, h:{height:.1f}"
    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"SVG visualization saved to: {svg_output_file}")
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")

def post_process_tags(data: Dict, features: List[Dict], predicted_tags: List[str]):
    def process_nodes(node_item: Dict, feature_index: int) -> int:
        if not node_item or "children" not in node_item:
            return feature_index
        node_dict = node_item.get("node", {})
        node_type = node_dict.get("type", "")
        if node_type == "GROUP":
            node_item["tag"] = "DIV"
            predicted_tags[feature_index] = "DIV"
        elif node_type == "TEXT":
            node_item["tag"] = "P"
            predicted_tags[feature_index] = "P"
        elif node_type in ["SVG", "VECTOR"] or node_item.get("name", "").startswith("ICON"):
            node_item["tag"] = "SVG"
            predicted_tags[feature_index] = "SVG"
        elif node_type == "LINE":
            node_item["tag"] = "HR"
            predicted_tags[feature_index] = "HR"
        elif (fills := node_dict.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills):
            node_item["tag"] = "IMG"
            predicted_tags[feature_index] = "IMG"
        else:
            node_item["tag"] = predicted_tags[feature_index]
        feature_index += 1
        for child in node_item.get("children", []):
            feature_index = process_nodes(child, feature_index)
        return feature_index

    def apply_post_processing_rules(node_item: Dict, body_width: float):
        if not node_item or "children" not in node_item:
            return
        children = node_item["children"]
        for i in range(len(children) - 1):
            if children[i].get("tag") == "P" and children[i + 1].get("tag") == "INPUT":
                children[i]["tag"] = "LABEL"
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
        for child in children:
            if child.get("tag") == "DIV":
                li_count = sum(1 for c in child.get("children", []) if c.get("tag") == "LI")
                if li_count >= 2:
                    child["tag"] = "UL"
        for child in children:
            if child.get("tag") == "DIV":
                form_elems = sum(1 for c in child.get("children", []) if c.get("tag") in ["INPUT", "BUTTON"])
                if form_elems >= 2:
                    child["tag"] = "FORM"
        for child in children:
            apply_post_processing_rules(child, body_width)

    process_nodes(data, 0)
    body_width = float(data.get("node", {}).get("width", 1000.0)) or 1000.0
    apply_post_processing_rules(data, body_width)

def predict_tags_tree_blstm(
    input_file: str,
    output_file: str,
    model_path: str,
    svg_file: Optional[str] = None
):
    global device
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
    root = extractor.extract_features(data, body_width, sequence_id)

    print("Predicting tags...")
    with torch.no_grad():
        logits_list = model.predict_all(root)
    predicted_indices = [torch.argmax(logits, dim=1).cpu().numpy()[0] for logits in logits_list]
    predicted_tags = label_encoder.inverse_transform(predicted_indices)

    nodes = root.get_nodes_depth_first()
    for node, tag in zip(nodes, predicted_tags):
        node.predicted_tag = tag

    features = [{"node_data": node.node_data} for node in nodes]
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
    predict_tags_tree_blstm(
        input_file="input.json",
        output_file="output.json",
        model_path="figma_tree_blstm_final_model.pt",
        svg_file="input.svg"
    )