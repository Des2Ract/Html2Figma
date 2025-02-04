import { FigmaNode } from "./figma_node.js";

export function extractStyles(element: Element): Record<string, string> {
  const styleObj: Record<string, string> = {}; // Use styleObj to store the styles

  const computedStyles = (
    element.ownerDocument?.defaultView ?? window
  ).getComputedStyle(element);

  // Loop through the computed styles
  for (let i = 0; i < computedStyles.length; i++) {
    const property = computedStyles[i]; // Get the property name
    const value = computedStyles.getPropertyValue(property); // Get the property value
    styleObj[property] = value; // Store the property and value in styleObj
  }

  return styleObj; // Return the collected styles
}

export function applyStylesToFigmaNode(element: Element): FigmaNode {
  const tagName = element.tagName.toLowerCase();
  const styles = extractStyles(element);

  return {
    type: "node",
    tagName,
    styles,
    children: [],
  };
}
