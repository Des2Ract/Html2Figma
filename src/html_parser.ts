import { FigmaNode, createFigmaNode } from "./figma_node";
import { extractStyles } from "./style_extractor";
import { applyFigmaType } from "./type_mapper";

// Parse HTML to Figma-like nodes
export function parseHTMLToFigmaNode(element: HTMLElement): FigmaNode {
  const styles = extractStyles(element);
  const node = createFigmaNode(element.tagName.toLowerCase(), styles);

  // Apply Figma type mapping based on both tag and styles
  applyFigmaType(node);

  // Recursively parse child elements
  for (const child of Array.from(element.children)) {
    const childNode = parseHTMLToFigmaNode(child as HTMLElement);
    node.children.push(childNode);
  }

  return node;
}
