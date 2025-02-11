import { FigmaNode, createFigmaNode } from './figma_node.js';
import {
  handleImageNode,
  handleLineNode,
  handlePictureNode,
  handleSvgNode,
  handleTextNode,
  handleVideoNode,
} from './FigmaComponentHandlers.js';

export function extractFigmaNode(element: Element): FigmaNode | null {
  // Skip empty or whitespace text nodes
  if (element.nodeType === Node.TEXT_NODE && (!element.nodeValue || element.nodeValue.trim() === '')) {
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
    (element.nodeValue.includes('function') ||
      element.nodeValue.includes('var ') ||
      element.nodeValue.includes('if (') ||
      element.nodeValue.includes('else {'))
  ) {
    return null;
  }

  // Skip nodes with specific tag names or attributes
  if (element.tagName === 'SCRIPT' || element.tagName === 'STYLE' || element.tagName === 'NOSCRIPT') {
    return null;
  }

  // TODO: TEXT NODE
  function isTextOnlyNode(element: Element): boolean {
    // Direct text node
    if (element.nodeType === Node.TEXT_NODE) {
      return true;
    }

    // Tags allowed to encapsulate simple text
    // prettier-ignore
    const allowedTextTags = new Set(['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B', 'U', 'I', 'STRONG', 'EM', 'MARK', 'SMALL', 'SUB', 'SUP', 'INS', 'DEL', 'CITE', 'Q', 'BLOCKQUOTE', 'CODE', 'VAR', 'PRE', 'SAMP', 'KBD', 'DFN', 'ABBR', 'SPAN', 'LABEL']);

    // Check if it's a non-grouping tag from allowedTextTags
    if (allowedTextTags.has(element.tagName)) {
      const nonEmptyTextChildren = Array.from(element.childNodes).filter(
        (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim() !== '',
      );
      return nonEmptyTextChildren.length === 1;
    }

    // Grouping tags are never simple text nodes
    return false;
  }

  if (isTextOnlyNode(element))
    return createFigmaNode(element.tagName ? element.tagName : 'txt', handleTextNode(element));

  // TODO: Hidden NODE
  const isHiddenNode = (element: HTMLElement): boolean =>
    getComputedStyle(element).display === 'none' ||
    getComputedStyle(element).visibility === 'hidden' ||
    parseFloat(getComputedStyle(element).opacity) === 0;

  if (isHiddenNode(element as HTMLElement)) return null;

  // TODO: IMAGE NODE
  if (element instanceof HTMLImageElement)
    return createFigmaNode(element.tagName ? element.tagName : 'txt', handleImageNode(element));

  // TODO: PICTURE NODE
  if (element instanceof HTMLPictureElement)
    return createFigmaNode(element.tagName ? element.tagName : 'txt', handlePictureNode(element));
  // TODO: VIDEO NODE
  if (element instanceof HTMLVideoElement)
    return createFigmaNode(element.tagName ? element.tagName : 'txt', handleVideoNode(element));

  // TODO: SVG NODE
  if (element instanceof SVGSVGElement) return createFigmaNode(element.tagName, handleSvgNode(element));

  // TODO: HR element
  if (element instanceof HTMLHRElement) return createFigmaNode(element.tagName, handleLineNode(element));

  // TODO: DIV/SPAN NODE

  // TODO: BUTTON NODE

  // TODO: A/LINK NODE

  // TODO: TABLE NODE

  // TODO: FORM NODE

  // TODO: INPUT NODE

  // Create and return a general Figma node
  return createFigmaNode(element.tagName ? element.tagName : 'txt', {} as any);
}
