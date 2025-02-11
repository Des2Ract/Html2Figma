import { FigmaNode } from './figma_node.js';
import { extractFigmaNode } from './FigmaNodeExtractor.js';

export function parseHTMLToFigmaNode(element: Element): FigmaNode | null {
  // Create the root Figma node
  const rootNode: FigmaNode | null = extractFigmaNode(element);
  if (rootNode == null) return null;
  // Recursively process valid child nodes
  if (
    rootNode.node.type === 'TEXT' ||
    rootNode.node.type === 'SVG' ||
    rootNode.tag === 'PICTURE'
  )
    return rootNode;
  element.childNodes.forEach((child) => {
    const childFigmaNode = parseHTMLToFigmaNode(child as Element);
    if (childFigmaNode != null) rootNode.children.push(childFigmaNode);
  });

  return rootNode;
}
