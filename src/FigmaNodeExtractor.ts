import { FigmaNode, createFigmaNode } from './figma_node.js';
import {
  handleButtonFormNode,
  handleDefaultNode,
  handleImageNode,
  handleLineNode,
  handleLinkNode,
  handlePictureNode,
  handleSvgNode,
  handleTextNode,
  handleVideoNode,
  handleBodyNode,
  handleInputNode,
} from './FigmaComponentHandlers.js';

export function extractFigmaNode(element: Element): FigmaNode | null {
  const TXTNODETAG = 'TXT';
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
    const allowedTextTags = new Set(['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B', 'U', 'I', 'STRONG', 'EM', 'MARK', 'SMALL', 'SUB', 'SUP', 'INS', 'DEL', 'CITE', 'Q', 'BLOCKQUOTE', 'CODE', 'VAR', 'PRE', 'SAMP', 'KBD', 'DFN', 'ABBR', 'SPAN', 'LABEL', 'LI']);

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
    return createFigmaNode(element.tagName ? element.tagName : TXTNODETAG, handleTextNode(element));

  // TODO: Hidden NODE
  const isHiddenNode = (element: HTMLElement): boolean =>
    getComputedStyle(element).display === 'none' ||
    getComputedStyle(element).visibility === 'hidden' ||
    parseFloat(getComputedStyle(element).opacity) === 0;

  if (isHiddenNode(element as HTMLElement)) return null;

  // TODO: IMAGE NODE / Handle Strokes
  if (element instanceof HTMLImageElement)
    return createFigmaNode(element.tagName ? element.tagName : TXTNODETAG, handleImageNode(element));

  // TODO: PICTURE NODE / Handle Strokes
  if (element instanceof HTMLPictureElement)
    return createFigmaNode(element.tagName ? element.tagName : TXTNODETAG, handlePictureNode(element));
  // TODO: VIDEO NODE / Handle Strokes
  if (element instanceof HTMLVideoElement)
    return createFigmaNode(element.tagName ? element.tagName : TXTNODETAG, handleVideoNode(element));

  // TODO: SVG NODE / Handle Strokes
  if (element instanceof SVGSVGElement) return createFigmaNode(element.tagName, handleSvgNode(element));

  // TODO: HR element
  if (element instanceof HTMLHRElement) return createFigmaNode(element.tagName, handleLineNode(element));

  // TODO: BUTTON/FORM NODE / Handle Strokes
  if (element instanceof HTMLButtonElement || element instanceof HTMLFormElement)
    return createFigmaNode(element.tagName, handleButtonFormNode(element));

  // TODO: A/LINK NODE
  if (element instanceof HTMLAnchorElement) return createFigmaNode(element.tagName, handleLinkNode(element));
  // TODO: Handle BODY
  if (element instanceof HTMLBodyElement) return createFigmaNode(element.tagName, handleBodyNode(element));

  // TODO: INPUT NODE
  if (element instanceof HTMLInputElement) return handleInputNode(element);

  // Default: return a Group Figma Node / Handle Strokes
  return createFigmaNode(element.tagName ? element.tagName : TXTNODETAG, handleDefaultNode(element));
}
