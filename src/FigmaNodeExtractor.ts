import { FigmaNode, createFigmaNode } from "./figma_node.js";
import {
  handleImageNode,
  handleSvgNode,
  handleTextNode,
} from "./FigmaComponentHandlers.js";

export function extractFigmaNode(element: Element): FigmaNode | null {
  //   return extractFigmaNode2(element);

  // Skip empty or whitespace text nodes
  if (
    element.nodeType === Node.TEXT_NODE &&
    (!element.nodeValue || element.nodeValue.trim() === "")
  ) {
    return null;
  }

  // Skip comment nodes
  if (element.nodeType === Node.COMMENT_NODE) {
    return null;
  }

  // Skip nodes containing JavaScript code or specific patterns
  if (
    element.nodeType === Node.TEXT_NODE &&
    element.nodeValue &&
    (element.nodeValue.includes("function") ||
      element.nodeValue.includes("var ") ||
      element.nodeValue.includes("if (") ||
      element.nodeValue.includes("else {"))
  ) {
    return null;
  }

  // Skip nodes with specific tag names or attributes
  if (
    element.tagName === "SCRIPT" ||
    element.tagName === "STYLE" ||
    element.tagName === "NOSCRIPT"
  ) {
    return null;
  }

  // TODO: TEXT NODE
  function isTextOnlyNode(element: Element) {
    // Direct text node
    if (element.nodeType === Node.TEXT_NODE) {
      return true;
    }

    // Check children for a single, non-empty text node
    const nonEmptyTextChildren = Array.from(element.childNodes).filter(
      (node) =>
        node.nodeType === Node.TEXT_NODE && node.textContent?.trim() !== ""
    );

    return nonEmptyTextChildren.length === 1;
  }

  if (isTextOnlyNode(element))
    return createFigmaNode(
      element.tagName ? element.tagName : "txt",
      handleTextNode(element)
    );

  // TODO: IMAGE NODE
  if (element instanceof HTMLImageElement)
    return createFigmaNode(
      element.tagName ? element.tagName : "txt",
      handleImageNode(element)
    );

  // TODO: PICTURE NODE

  // TODO: VIDEO NODE

  // TODO: Hidden NODE

  // TODO: SVG NODE
  if (element instanceof SVGSVGElement)
    return createFigmaNode(element.tagName, handleSvgNode(element));

  // Create and return a general Figma node
  return createFigmaNode(element.tagName ? element.tagName : "txt", {} as any);
}
