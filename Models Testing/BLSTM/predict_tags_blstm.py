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

# Disable oneDNN optimizations (if using TensorFlow elsewhere)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class OptimizedFigmaBLSTM(nn.Module):
    """Memory-optimized Bidirectional LSTM model."""
    
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
        
        # BatchNorm1d over the concatenated forward+backward hidden size
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Pack padded sequence (lengths must be on CPU)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Apply dropout
        output = self.dropout(output)
        
        # BatchNorm1d expects (N, features)
        if batch_size > 1:
            # reshape to (batch_size * seq_len, hidden_dim*2)
            output_reshaped = output.reshape(-1, output.size(-1))
            output_reshaped = self.batch_norm(output_reshaped)
            output = output_reshaped.reshape(batch_size, seq_len, -1)
        else:
            # For batch_size == 1, reshape to (seq_len, hidden_dim*2)
            output_reshaped = output.view(-1, output.size(-1))
            output_normed = self.batch_norm(output_reshaped)
            output = output_normed.view(batch_size, seq_len, -1)
        
        # Final classification layer applied at each time step
        logits = self.fc(output)  # shape: (batch_size, seq_len, num_classes)
        return logits


class FigmaHTMLFeatureExtractor:
    def __init__(
        self,
        semantic_model_name: str = 'all-MiniLM-L6-v2',
        node_type_embedding_dim: int = 50
    ):
        # Sentence‐Transformer for text & node-name embeddings
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.text_embedding_dim = 384  # for 'all-MiniLM-L6-v2'
        self.node_name_embedding_dim = 384  # same dimension if we use the same model

        # Node‐type embedding
        self.node_types = self._get_node_types()
        self.node_type_to_idx = {node_type: idx for idx, node_type in enumerate(self.node_types)}
        self.node_type_embedding_dim = node_type_embedding_dim
        self.node_type_embedding_layer = nn.Embedding(len(self.node_types), self.node_type_embedding_dim)

        # Tag mapping and cleaning
        self.tag_mapping = self._get_tag_mapping()
        self.custom_tag_removal_pattern = self._get_custom_tag_removal_pattern()
        self.default_tag = "DIV"
        self.icon_like_node_types = {"VECTOR", "INSTANCE", "COMPONENT", "SHAPE", "SVG_ICON"}

        # Statistics counters
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
        """Clean and map a raw HTML tag to a canonical form."""
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
        """Determine BIOES-style label for a tag (we only use B_CONTAINER vs. others)."""
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
        total_siblings: int = 1
    ) -> List[Dict]:
        """
        Recursively extract features from a single node and its children.
        Returns a flat list of dicts with keys: 'feature_vector', 'tag', 'node_data'.
        """
        features_and_labels_list = []
        node_dict = node_data_item.get("node", {})
        raw_tag = node_data_item.get("tag", "UNK").upper()

        # Determine base tag, then BIOES label
        has_children = bool(node_data_item.get("children"))
        base_tag = self.clean_and_map_tag(raw_tag)
        bioes_label = self.determine_bioes_label(base_tag)

        # Node‐type embedding
        node_type_str = node_dict.get("type", "UNKNOWN_TYPE")
        self.stats["unique_node_types"].add(node_type_str)
        node_type_idx = self.node_type_to_idx.get(node_type_str, self.node_type_to_idx["UNKNOWN_TYPE"])
        node_type_emb = (
            self.node_type_embedding_layer(torch.tensor(node_type_idx, dtype=torch.long))
            .detach()
            .cpu()
            .numpy()
        )

        # Text embedding (only if TEXT node with characters)
        text_content = node_dict.get("characters", "").strip()
        if node_type_str == "TEXT" and text_content:
            text_emb = self.semantic_model.encode(text_content)
        else:
            text_emb = np.zeros(self.text_embedding_dim)

        # Node‐name embedding (if icon‐like or name contains “icon”)
        node_name = node_data_item.get("name", "").strip()
        if node_name and (node_type_str in self.icon_like_node_types or "icon" in node_name.lower()):
            node_name_emb = self.semantic_model.encode(node_name)
        else:
            node_name_emb = np.zeros(self.node_name_embedding_dim)

        # Numerical & structural features
        eps = 1e-6
        node_width = float(node_dict.get("width", 0))
        node_height = float(node_dict.get("height", 0))
        aspect_ratio = node_width / (node_height + eps) if node_height > 0 else 0.0
        normalized_width = node_width / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_height = node_height / (parent_node_height + eps) if parent_node_height and parent_node_height > 0 else 0.0

        x_position = float(node_dict.get("x", 0))
        y_position = float(node_dict.get("y", 0))
        normalized_x = x_position / (current_body_width + eps) if current_body_width > 0 else 0.0
        normalized_y = y_position / (parent_node_height + eps) if parent_node_height and parent_node_height > 0 else 0.0

        normalized_depth = min(depth / 20.0, 1.0)
        normalized_position = position_in_siblings / (total_siblings + eps)

        # Background color RGBA (take first fill if present)
        bg_color = [0.0, 0.0, 0.0, 0.0]
        fills = node_dict.get("fills", [])
        if fills and isinstance(fills, list) and len(fills) > 0:
            color = fills[0].get("color", {})
            bg_color = [
                float(color.get("r", 0.0)),
                float(color.get("g", 0.0)),
                float(color.get("b", 0.0)),
                float(color.get("a", 0.0)),
            ]

        font_size = float(node_dict.get("fontSize", 0.0)) / 100.0
        flex_direction = 1.0 if node_dict.get("flexDirection", "") == "column" else 0.0

        # Combine into one feature vector
        feature_vector = np.concatenate([
            node_type_emb,
            text_emb,
            node_name_emb,
            np.array([
                normalized_width, normalized_height, aspect_ratio,
                normalized_x, normalized_y,
                normalized_depth, normalized_position,
                *bg_color, font_size, flex_direction
            ], dtype=np.float32)
        ])

        features_and_labels_list.append({
            "feature_vector": feature_vector.astype(np.float32),
            "tag": bioes_label,
            "node_data": node_data_item
        })

        self.stats["nodes_processed"] += 1

        # Recurse into children
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
                    total_siblings=total_children
                )
                features_and_labels_list.extend(child_features)

            # After a container is done, append an E_CONTAINER vector (all zeros)
            if bioes_label == "B_CONTAINER":
                zero_vector = np.zeros_like(feature_vector, dtype=np.float32)
                features_and_labels_list.append({
                    "feature_vector": zero_vector,
                    "tag": "E_CONTAINER",
                    "node_data": None
                })

        # At the end of the root node, append an E_WEBSITE vector (all ones)
        if depth == 0:
            one_vector = np.ones_like(feature_vector, dtype=np.float32)
            features_and_labels_list.append({
                "feature_vector": one_vector,
                "tag": "E_WEBSITE",
                "node_data": None
            })

        return features_and_labels_list


def load_model_and_metadata(model_path: str, device: str = None):
    """Load the trained BLSTM model and associated metadata from a checkpoint."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['model_config']
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']  # e.g. a sklearn LabelEncoder

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
    """Generate a random RGBA color string for SVG fills."""
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
    Draw bounding boxes and predicted tags on a copy of an existing SVG file.
    """
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"

    # Copy original SVG
    shutil.copy2(svg_input_file, svg_output_file)

    # Parse the SVG
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()

    # Add a <style> block for our annotation classes
    style_el = etree.SubElement(root, 'style')
    style_el.text = """
        .tag-box { stroke: #000000; stroke-width: 1; fill-opacity: 0.3; }
        .tag-text { font-family: Arial; font-size: 10px; }
        .tag-label { fill: white; stroke: #000000; stroke-width: 0.5; rx: 3; ry: 3; }
    """

    # A group to hold all tag annotations
    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    tag_colors = {}

    for feat_dict, tag in zip(features, predicted_tags):
        node_data_item = feat_dict.get("node_data")
        if not node_data_item or not node_data_item.get("node"):
            # Skip E_CONTAINER or E_WEBSITE entries
            continue

        node = node_data_item["node"]
        x, y = float(node.get("x", 0.0)), float(node.get("y", 0.0))
        width, height = float(node.get("width", 50.0)), float(node.get("height", 50.0))

        if tag not in tag_colors:
            tag_colors[tag] = generate_random_color()
        color = tag_colors[tag]

        grp = etree.SubElement(tag_group, 'g')

        # Draw bounding rectangle
        etree.SubElement(grp, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "class": "tag-box",
            "fill": color,
            "stroke": "black",
            "stroke-width": "1"
        })

        # Background for label text
        label_w = max(80, len(tag) * 7)
        label_h = 40
        etree.SubElement(grp, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(label_w),
            "height": str(label_h),
            "rx": "3",
            "ry": "3",
            "fill": "white",
            "fill-opacity": "0.7",
            "stroke": "black",
            "stroke-width": "0.5"
        })

        # Tag name
        etree.SubElement(grp, 'text', {
            "x": str(x + 3),
            "y": str(y + 12),
            "class": "tag-text",
            "fill": "black"
        }).text = tag

        # Coordinates and size text
        etree.SubElement(grp, 'text', {
            "x": str(x + 3),
            "y": str(y + 24),
            "class": "tag-text",
            "fill": "black"
        }).text = f"x:{x:.1f}, y:{y:.1f}"
        etree.SubElement(grp, 'text', {
            "x": str(x + 3),
            "y": str(y + 36),
            "class": "tag-text",
            "fill": "black"
        }).text = f"w:{width:.1f}, h:{height:.1f}"

    # Write out the updated SVG
    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"SVG visualization saved to: {svg_output_file}")
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")


def post_process_tags(data: Dict, features: List[Dict], predicted_tags: List[str]):
    """
    Post-process predicted tags: apply hand‐crafted rules (e.g., turn P+INPUT into LABEL, group LIST items, etc.).
    """

    def process_nodes(node_item: Dict, feature_index: int) -> int:
        """
        Recursively assign predicted_tags back to each node_data_item, and apply a few simple rules.
        Returns updated feature_index.
        """
        if not node_item or "children" not in node_item:
            return feature_index

        # If this node is a GROUP, force DIV; if TEXT, force P; if SVG/VECTOR or name startsWith ICON, force SVG; etc.
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
            # Default: use the model’s predicted tag
            node_item["tag"] = predicted_tags[feature_index]

        feature_index += 1

        # Recurse on children
        for child in node_item.get("children", []):
            feature_index = process_nodes(child, feature_index)

        return feature_index

    def apply_post_processing_rules(node_item: Dict, body_width: float):
        """
        After tags have been assigned, apply some high-level rules:
          - P followed by INPUT ⇒ first becomes LABEL
          - DIV at (0,0) with width ~ body_width & small height ⇒ NAVBAR
          - DIV with ≥2 LI children ⇒ UL
          - DIV with ≥2 form elements ⇒ FORM
        """
        if not node_item or "children" not in node_item:
            return

        children = node_item["children"]
        # 1. P + INPUT ⇒ LABEL
        for i in range(len(children) - 1):
            if children[i].get("tag") == "P" and children[i + 1].get("tag") == "INPUT":
                children[i]["tag"] = "LABEL"

        # 2. NAVBAR detection
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

        # 3. DIV with ≥2 LI ⇒ UL
        for child in children:
            if child.get("tag") == "DIV":
                li_count = sum(1 for c in child.get("children", []) if c.get("tag") == "LI")
                if li_count >= 2:
                    child["tag"] = "UL"

        # 4. DIV with ≥2 form elements (INPUT or BUTTON) ⇒ FORM
        for child in children:
            if child.get("tag") == "DIV":
                form_elems = sum(1 for c in child.get("children", []) if c.get("tag") in ["INPUT", "BUTTON"])
                if form_elems >= 2:
                    child["tag"] = "FORM"

        # Recurse
        for child in children:
            apply_post_processing_rules(child, body_width)

    # 1) Assign each predicted tag back to node_data_item (with some overrides)
    process_nodes(data, 0)

    # 2) Apply rules based on children structure
    body_width = float(data.get("node", {}).get("width", 1000.0)) or 1000.0
    apply_post_processing_rules(data, body_width)


def predict_tags_blstm(
    input_file: str,
    output_file: str,
    model_path: str,
    svg_file: Optional[str] = None
):
    """
    Main pipeline: load JSON, extract features, run BLSTM, post‐process, save JSON, and optionally draw SVG annotations.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model + metadata
    model, label_encoder, input_dim, model_config = load_model_and_metadata(model_path, device)
    print("Model and metadata loaded successfully.")

    # Initialize feature extractor
    extractor = FigmaHTMLFeatureExtractor(
        semantic_model_name='all-MiniLM-L6-v2',
        node_type_embedding_dim=50
    )

    # Load input JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sequence_id = os.path.basename(input_file).replace(".json", "")
    body_width = float(data.get("node", {}).get("width", 1000.0)) or 1000.0

    # 1) Extract features from every node (flat list)
    print("Extracting features...")
    features = extractor.extract_features(
        node_data_item=data,
        current_body_width=body_width,
        sequence_id=sequence_id
    )

    # 2) Stack feature vectors into a single (seq_len, feature_dim) array
    feature_vectors = np.stack([f["feature_vector"] for f in features]).astype(np.float32)
    seq_len = feature_vectors.shape[0]

    # Convert to torch tensor, add batch dimension: (1, seq_len, input_dim)
    padded_features = torch.from_numpy(feature_vectors).unsqueeze(0).to(device)

    # Lengths = [seq_len] for pack_padded_sequence (keep on CPU)
    lengths = torch.tensor([seq_len], dtype=torch.long)

    # 3) Run through the BLSTM
    print("Predicting tags...")
    with torch.no_grad():
        outputs = model(padded_features, lengths)  # shape: (1, seq_len, num_classes)
        predicted_indices = torch.argmax(outputs, dim=2).cpu().numpy().flatten()
        predicted_tags = label_encoder.inverse_transform(predicted_indices)

    # 4) Post-process tags & assign them back into the JSON structure
    print("Post-processing tags...")
    post_process_tags(data, features, list(predicted_tags))

    # 5) Save the updated JSON to output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Processed JSON saved to: {output_file}")

    # 6) If an SVG was provided, generate a tagged copy
    if svg_file:
        print("Drawing annotations on SVG...")
        draw_tags_on_svg_file(data, features, list(predicted_tags), svg_file)

    # 7) Print statistics
    print("\nProcessing Statistics:")
    print(f"  Nodes processed: {extractor.stats['nodes_processed']}")
    print(f"  Unique node types seen: {len(extractor.stats['unique_node_types'])}")
    print(f"  Total tag mappings applied: {sum(extractor.stats['tag_mappings'].values())}")


if __name__ == "__main__":
    # Example (adjust paths as needed):
    predict_tags_blstm(
        input_file="input.json",
        output_file="output.json",
        model_path="figma_blstm_final_model.pt",
        svg_file="input.svg"
    )
