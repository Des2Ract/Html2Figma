// Helper function for basic tag-to-Figma type mapping
export function mapTagToNodeType(tag) {
    switch (tag.toLowerCase()) {
        case "div":
            return "FRAME"; // Div maps to a Frame (flex container, etc.)
        case "p":
            return "TEXT"; // Paragraph maps to Text node
        case "img":
            return "IMAGE"; // Image tag maps to Image node
        case "button":
            return "BUTTON"; // Button tag maps to Button node
        default:
            return "GROUP"; // Default mapping to Group node
    }
}
// Mapping based on additional styles
export function mapStylesToFigmaType(styles) {
    if (styles["display"] === "flex") {
        return "FRAME"; // Flex container in HTML maps to Frame in Figma
    }
    if (styles["position"] === "absolute") {
        return "FRAME"; // Positioned elements may map to a Frame in Figma
    }
    if (styles["display"] === "inline-block") {
        return "GROUP"; // Inline-block elements might be grouped in Figma
    }
    return "GROUP"; // Default to Group if no special styles
}
// Map both the tag and the styles to determine Figma node type
export function mapToFigmaType(tag, styles) {
    const tagType = mapTagToNodeType(tag);
    const styleType = mapStylesToFigmaType(styles);
    // If the style mapping provides a specific type (e.g., Frame), use it
    return styleType !== "GROUP" ? styleType : tagType;
}
// Apply Figma type to a node based on its tag and styles
export function applyFigmaType(node) {
    const figmaType = mapToFigmaType(node.tagName, node.styles);
    node.type = figmaType;
}
