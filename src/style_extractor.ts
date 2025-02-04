import { FigmaNode } from "./figma_node";

export function extractStyles(element: HTMLElement): Record<string, string> {
  const computedStyles = getComputedStyle(element);
  const styleObj: Record<string, string> = {};

  for (const prop of computedStyles) {
    styleObj[prop] = computedStyles.getPropertyValue(prop);
  }

  return styleObj;
}

export function applyStylesToFigmaNode(element: HTMLElement): FigmaNode {
  const tagName = element.tagName.toLowerCase();
  const styles = extractStyles(element);
  return {
    type: "node",
    tagName,
    styles,
    children: [],
  };
}
