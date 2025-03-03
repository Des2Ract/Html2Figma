import json

def determine_tag(figma_node, is_root=False):
    """
    Determine a tag for the node. For the root node we return "BODY".
    Otherwise, if the node type is TEXT we return "TXT", if the name contains "header"
    we return "HEADER", if the node is a link (has linkUnfurlData) we return "A",
    if it's a FRAME or GROUP we return "DIV". Otherwise default to "UNK".
    """
    if is_root:
        return "BODY"
    node_type = figma_node.get("type", "").upper()
    name = figma_node.get("name", "").lower()
    
    if node_type == "TEXT":
        return "TXT"
    elif "header" in name:
        return "HEADER"
    elif figma_node.get("linkUnfurlData"):
        return "A"
    elif node_type in ["FRAME", "GROUP"]:
        return "DIV"
    else:
        return "UNK"

def convert_node(figma_node, offset_x, offset_y, is_root=False):
    """
    Recursively convert a Figma node into the new structure.
    
    - Uses absoluteBoundingBox (adjusted by the offset) for x, y, width, height.
    - If node type is TEXT, it adds font size, fontFamily, and textAlign.
    - For RECTANGLE nodes, it adds fills and radius properties.
    - For FRAME and GROUP nodes it maps fills/backgrounds and effects.
    - Recursively converts children.
    """
    new_node = {}
    new_node["tag"] = "UNK"
    
    # Build node details
    node_data = {}
    node_data["type"] = figma_node.get("type", "UNK")
    
    # Adjust coordinates based on absoluteBoundingBox (if present)
    bbox = figma_node.get("absoluteBoundingBox", {})
    x = bbox.get("x", 0) - offset_x
    y = bbox.get("y", 0) - offset_y
    node_data["x"] = x
    node_data["y"] = y
    node_data["width"] = bbox.get("width", 0)
    node_data["height"] = bbox.get("height", 0)
    
    # Add extra properties based on the type of the node
    if figma_node.get("type") == "TEXT":
        node_data["characters"] = figma_node.get("characters", "")
        if "style" in figma_node:
            style = figma_node["style"]
            node_data["fontSize"] = style.get("fontSize")
            node_data["fontName"] = {
                "family": style.get("fontFamily"),
                "style": style.get("fontStyle")
            }
            node_data["textAlignHorizontal"] = style.get("textAlignHorizontal")
    elif figma_node.get("type") == "RECTANGLE":
        node_data["fills"] = figma_node.get("fills", [])
        # If radius properties exist they might be directly on the node
        node_data["topLeftRadius"] = figma_node.get("topLeftRadius", 0)
        node_data["topRightRadius"] = figma_node.get("topRightRadius", 0)
        node_data["bottomLeftRadius"] = figma_node.get("bottomLeftRadius", 0)
        node_data["bottomRightRadius"] = figma_node.get("bottomRightRadius", 0)
        if "strokes" in figma_node:
            node_data["strokes"] = figma_node.get("strokes")
    elif figma_node.get("type") == "FRAME":
        node_data["fills"] = figma_node.get("fills", [])
        # In our sample, FRAME nodes include radius info (here defaulted to 0)
        node_data["topLeftRadius"] = 0
        node_data["topRightRadius"] = 0
        node_data["bottomLeftRadius"] = 0
        node_data["bottomRightRadius"] = 0
    elif figma_node.get("type") == "GROUP":
        # For groups, we use backgrounds and effects as in the sample output.
        node_data["backgrounds"] = figma_node.get("fills", [])
        node_data["effects"] = figma_node.get("effects", [])
    
    # If the node has link data, include it
    if "linkUnfurlData" in figma_node:
        node_data["linkUnfurlData"] = figma_node["linkUnfurlData"]
    
    new_node["node"] = node_data
    
    # Process children recursively
    new_children = []
    for child in figma_node.get("children", []):
        converted_child = convert_node(child, offset_x, offset_y, is_root=False)
        new_children.append(converted_child)
    new_node["children"] = new_children
    
    return new_node

def convert_figma_json(figma_json):
    """
    Entry point to convert the Figma JSON.
    
    The conversion starts at the "document" object within the first node.
    The x and y from the document’s absoluteBoundingBox become the offset,
    so that the root is at (0, 0) and all children are shifted accordingly.
    """
    nodes = figma_json.get("nodes", {})
    if not nodes:
        raise ValueError("No nodes found in Figma JSON.")
    # Get the first node's document (e.g., key like "1:520")
    first_key = next(iter(nodes))
    document = nodes[first_key].get("document", {})
    
    bbox = document.get("absoluteBoundingBox", {})
    offset_x = bbox.get("x", 0)
    offset_y = bbox.get("y", 0)
    
    return convert_node(document, offset_x, offset_y, is_root=True)

# Example usage:
if __name__ == "__main__":
    # Load the input Figma JSON from a file (for example, "figma_input.json")
    with open("figma_input.json", "r") as infile:
        figma_data = json.load(infile)
    
    # Convert the JSON structure
    converted = convert_figma_json(figma_data)
    
    # Save the converted structure to a new JSON file (for example, "converted_output.json")
    with open("converted_output.json", "w") as outfile:
        json.dump(converted, outfile, indent=2)
