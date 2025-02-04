import { createFigmaNode } from "./figma_node.js";
import { extractStyles } from "./style_extractor.js";
import { applyFigmaType } from "./type_mapper.js";
import { JSDOM } from "jsdom";

// Parse HTML to Figma-like nodes
export function parseHTMLToFigmaNode(element: HTMLElement) {
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
