import json

def format_color(color_dict):
    """Format color object into short readable string"""
    if 'a' in color_dict:
        return f"rgba({color_dict['r']:.2f},{color_dict['g']:.2f},{color_dict['b']:.2f},{color_dict['a']:.2f})"
    else:
        return f"rgb({color_dict['r']:.2f},{color_dict['g']:.2f},{color_dict['b']:.2f})"

def get_node_summary(node):
    """Get important node information in a compact format"""
    info_parts = []
    
    # Position and size
    x, y = node.get('x', 0), node.get('y', 0)
    w, h = node.get('width', 0), node.get('height', 0)
    info_parts.append(f"pos:({x},{y}) size:{w}x{h}")
    
    # Type-specific info
    if node['type'] == 'TEXT':
        text = node.get('characters', '')[:30] + ('...' if len(node.get('characters', '')) > 200 else '')
        font_family = node.get('fontName', {}).get('family', 'Unknown')
        font_size = node.get('fontSize', 'Unknown')
        info_parts.append(f'text:"{text}" font:{font_family} size:{font_size}')
    
    elif node['type'] == 'SVG':
        info_parts.append("svg")
    
    elif node['type'] in ['RECTANGLE', 'FRAME']:
        # Check for border radius
        radii = [node.get(f'{corner}Radius', 0) for corner in ['topLeft', 'topRight', 'bottomLeft', 'bottomRight']]
        if any(r > 0 for r in radii):
            if all(r == radii[0] for r in radii):
                info_parts.append(f"radius:{radii[0]}")
            else:
                info_parts.append(f"radius:TL{radii[0]}/TR{radii[1]}/BL{radii[2]}/BR{radii[3]}")
    
    # Fill color (first fill only)
    if 'fills' in node and node['fills'] and node['fills'][0]['type'] == 'SOLID':
        fill_color = format_color(node['fills'][0]['color'])
        info_parts.append(f"fill:{fill_color}")
    
    # Stroke color (first stroke only)
    if 'strokes' in node and node['strokes'] and node['strokes'][0]['type'] == 'SOLID':
        stroke_color = format_color(node['strokes'][0]['color'])
        stroke_weight = node.get('strokeWeight', 1)
        info_parts.append(f"stroke:{stroke_color}({stroke_weight}px)")
    
    return " | ".join(info_parts)

def print_element(element, indent=0):
    """Print element tree in compact single-line format"""
    spaces = "  " * indent
    tag = element.get('tag', 'UNKNOWN')
    node = element.get('node', {})
    node_type = node.get('type', 'UNKNOWN')
    
    # Get summary info
    summary = get_node_summary(node)
    
    # Print single line with all info
    print(f"{spaces}{tag} ({node_type}) - {summary}")
    
    # Print children
    children = element.get('children', [])
    for child in children:
        print_element(child, indent + 1)

def main():
    file_path = "../json_data/figmaTree_1.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return
    
    print("FIGMA TREE STRUCTURE")
    print("=" * 50)
    
    # Print the tree
    print_element(data)
    
    # Count total elements
    def count_elements(element):
        return 1 + sum(count_elements(child) for child in element.get('children', []))
    
    total = count_elements(data)
    print(f"\nTotal elements: {total}")

if __name__ == "__main__":
    main()