import {
  FigmaNode,
  createFigmaNode,
  SvgNode,
  LayerNode,
} from "./figma_node.js";
import { getFigmaRGB } from "./utils.js";

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

  // TODO: Hidden NODE

  // TODO: PICTURE NODE

  // TODO: IMAGE NODE

  // TODO: PICTURE NODE

  // TODO: VIDEO NODE

  // TODO: SVG NODE
  if (element instanceof SVGSVGElement) {
    return createFigmaNode(element.tagName, handleSvgNode(element));
  }

  // Create and return a general Figma node
  return createFigmaNode(element.tagName ? element.tagName : "txt", {} as any);
}

export function handleTextNode(element: Element): Partial<TextNode> {
  const parent = element.parentElement as Element;

  const computedStyles = getComputedStyle(
    element instanceof Element ? element : parent
  );

  const range = document.createRange();
  range.selectNode(element);
  const fastClone = (data: any) =>
    typeof data === "symbol" ? null : JSON.parse(JSON.stringify(data));
  const rect = fastClone(range.getBoundingClientRect());
  let x = Math.round(rect.left);
  let y = Math.round(rect.top);
  let width = Math.round(rect.width);
  let height = Math.round(rect.height);

  const fills: SolidPaint[] = [];
  let rgb = getFigmaRGB(computedStyles.color);

  if (rgb) {
    fills.push({
      type: "SOLID",
      color: {
        r: rgb.r,
        g: rgb.g,
        b: rgb.b,
      },
      blendMode: "NORMAL",
      visible: true,
      opacity: rgb.a || 1,
    } as SolidPaint);
  }

  const textnode: Partial<TextNode> = {
    type: "TEXT",
    characters: (element.textContent as string).trim(),
    x: x,
    y: y,
    width: width,
    height: height,
    textAlignHorizontal: "LEFT",
    textAlignVertical: "CENTER",
    fontSize: parseFloat(computedStyles.fontSize),
    fontName: {
      family: computedStyles.fontFamily,
      style: computedStyles.fontStyle,
    },
    fills: fills,
  };
  return textnode;
}

export function handleSvgNode(element: Element) {
  const rect = element.getBoundingClientRect();

  const svgNode: Partial<LayerNode> = {
    type: "SVG",
    svg: element.outerHTML,
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
  };

  return svgNode;
}
