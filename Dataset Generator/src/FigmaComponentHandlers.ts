import { LayerNode } from './figma_node.js';
import { getBorder, getFigmaRGB, parseBoxShadow } from './utils.js';
import { FigmaNode, createFigmaNode } from './figma_node.js';


/**
 * Converts an HTML text element to a Figma TextNode by extracting computed styles,
 * positioning, typography properties, text alignment, decorations, and effects
 */
export function handleTextNode(element: Element): Partial<TextNode> {
  const parent = element.parentElement as Element;

  const computedStyles = getComputedStyle(element instanceof Element ? element : parent);

  const range = document.createRange();
  range.selectNode(element);
  const fastClone = (data: any) => (typeof data === 'symbol' ? null : JSON.parse(JSON.stringify(data)));
  const rect = fastClone(range.getBoundingClientRect());
  let x = Math.round(rect.left);
  let y = Math.round(rect.top);
  let width = Math.round(rect.width);
  let height = Math.round(rect.height);

  const fills: SolidPaint[] = [];
  let rgb = getFigmaRGB(computedStyles.color);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: {
        r: rgb.r,
        g: rgb.g,
        b: rgb.b,
      },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  const mapTextAlignHorizontal = (
    cssAlign: string | undefined,
  ): 'LEFT' | 'RIGHT' | 'CENTER' | 'JUSTIFIED' | undefined =>
    (
      ({
        left: 'LEFT',
        right: 'RIGHT',
        center: 'CENTER',
        justify: 'JUSTIFIED',
      }) as const
    )[cssAlign as 'left' | 'right' | 'center' | 'justify'];

  const mapTextAlignVertical = (cssAlign: string | undefined): 'TOP' | 'CENTER' | 'BOTTOM' | undefined =>
    (({ top: 'TOP', middle: 'CENTER', bottom: 'BOTTOM' }) as const)[cssAlign as 'top' | 'middle' | 'bottom'];

  const mapTextDecoration = (textDecoration: string): TextDecoration =>
    textDecoration.includes('underline')
      ? ('UNDERLINE' as TextDecoration)
      : textDecoration.includes('line-through')
        ? ('LINE_THROUGH' as TextDecoration)
        : ('NONE' as TextDecoration);

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const textnode: Partial<TextNode> = {
    type: 'TEXT',
    characters: (element.textContent as string).trim(),
    x: x,
    y: y,
    width: width,
    height: height,
    textAlignHorizontal: mapTextAlignHorizontal(computedStyles.textAlign),
    textAlignVertical: mapTextAlignVertical(computedStyles.verticalAlign),
    fontSize: parseFloat(computedStyles.fontSize),
    fontName: {
      family: computedStyles.fontFamily.replace(/['"]/g, ''),
      style: computedStyles.fontStyle,
    },
    fills: fills,
    textDecoration: mapTextDecoration(computedStyles.textDecoration),
    effects: shadow ? [shadow] : [],
  };
  return textnode;
}


/**
 * Converts an SVG element to a Figma LayerNode by serializing the SVG content,
 * extracting positioning, borders/strokes, and visual effects like shadows
 */
export function handleSvgNode(element: Element): Partial<LayerNode> {
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);

  if (element instanceof SVGElement && !element.hasAttribute('xmlns')) {
    element.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  }

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const svgString = new XMLSerializer().serializeToString(element).replace(/"/g, "'");

  const svgNode: Partial<LayerNode> = {
    type: 'SVG',
    svg: svgString,
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    strokes: ((borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[]).filter(
      (stroke) => stroke !== null,
    ) as Paint[],
    strokeWeight: borderData?.strokeWeight || 0,
    dashPattern: borderData?.dashPattern || [],
    effects: shadow ? [shadow] : [],
  };

  return svgNode;
}


/**
 * Converts an HTML image element to a Figma RectangleNode by extracting the image URL,
 * positioning, border radius, scaling mode, borders, and visual effects
 */
export function handleImageNode(element: Element): Partial<RectangleNode> {
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);
  const url = (element as HTMLImageElement).src as string;

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const fills = [
    {
      url: url as string,
      type: 'IMAGE',
      scaleMode: computedStyles.objectFit === 'contain' ? 'FIT' : 'FILL',
      imageHash: null,
      flexDirection: computedStyles.flexDirection || 'none',
    } as ImagePaint,
  ] as ImagePaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);

  const imageNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    strokes: (borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[],
    strokeWeight: borderData?.strokeWeight || 0,
    dashPattern: borderData?.dashPattern || [],
    effects: shadow ? [shadow] : [],
  };

  return imageNode;
}


/**
 * Converts an HTML picture element to a Figma RectangleNode by extracting the source URL
 * from srcset, positioning, border radius, scaling mode, borders, and visual effects
 */
export function handlePictureNode(element: Element): Partial<RectangleNode> {
  const source = element.querySelector('source')?.srcset.split(/[,\s]+/g)[0];
  const formatUrl = (url: string) =>
    url?.trim()?.replace(/^\/\//, 'https://')?.replace(/^\//, `https://${location.host}/`) || '';

  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);
  const url = (element as HTMLImageElement).src as string;

  const fills = [
    {
      url: source ? (formatUrl(source) ? formatUrl(source) : null) : null,
      type: 'IMAGE',
      scaleMode: computedStyles.objectFit === 'contain' ? 'FIT' : 'FILL',
      imageHash: null,
      flexDirection: computedStyles.flexDirection || 'none',
    } as ImagePaint,
  ] as ImagePaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const pictureNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    strokes: (borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[],
    strokeWeight: borderData?.strokeWeight || 0,
    dashPattern: borderData?.dashPattern || [],
    effects: shadow ? [shadow] : [],
  };

  return pictureNode;
}


/**
 * Converts an HTML video element to a Figma RectangleNode by extracting video URL
 * from src or source elements, positioning, border radius, scaling mode, and effects
 */
export function handleVideoNode(element: Element): Partial<RectangleNode> {
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);
  const videoElement = element as HTMLVideoElement;

  // Extract video URL
  let url = videoElement.src || '';
  if (!url) {
    const sourceElement = videoElement.querySelector('source');
    url = sourceElement?.src || videoElement.poster || '';
  }

  const fills = [
    {
      url: url,
      type: 'VIDEO',
      scaleMode: computedStyles.objectFit === 'contain' ? 'FIT' : 'FILL',
      videoHash: null,
    } as VideoPaint,
  ] as VideoPaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);
  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const videoNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    strokes: (borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[],
    strokeWeight: borderData?.strokeWeight || 0,
    dashPattern: borderData?.dashPattern || [],
    effects: shadow ? [shadow] : [],
  };

  return videoNode;
}


/**
 * Converts an HTML hr (horizontal rule) element to a Figma LineNode by extracting
 * positioning, dimensions, background color fills, and visual effects
 */
export function handleLineNode(element: Element): Partial<LineNode> {
  const el = element as HTMLHRElement;
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(el);

  const fills: SolidPaint[] = [];
  let rgb = getFigmaRGB(computedStyles.backgroundColor);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: {
        r: rgb.r,
        g: rgb.g,
        b: rgb.b,
      },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const lineNode: Partial<LineNode> = {
    type: 'LINE',
    x: el.offsetLeft,
    y: el.offsetTop,
    width: parseFloat(computedStyles.width),
    height: parseFloat(computedStyles.height),
    fills: fills,
    effects: shadow ? [shadow] : [],
  };

  return lineNode;
}


/**
 * Converts various HTML input elements to appropriate Figma nodes based on input type:
 * - checkbox: RectangleNode
 * - radio: EllipseNode  
 * - button/submit/reset: RectangleNode with child TextNode
 * - text/email/search/number: RectangleNode with optional placeholder TextNode
 */
export function handleInputNode(element: Element): FigmaNode {
  const inputEl = element as HTMLInputElement;
  const inputType = inputEl.type.toLowerCase();

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);

  const textValue = ((element as HTMLInputElement).value || (element as HTMLInputElement).placeholder)?.trim();

  const fills: SolidPaint[] = [];

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  if (inputType == 'checkbox') {
    const checkboxNode: Partial<RectangleNode> = {
      type: 'RECTANGLE',
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      fills: fills,
      effects: shadow ? [shadow] : [],
    };
    return createFigmaNode('INPUT', checkboxNode);
  }

  if (inputType == 'radio') {
    const radioNode: Partial<EllipseNode> = {
      type: 'ELLIPSE',
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      fills: fills,
      effects: shadow ? [shadow] : [],
    };
    return createFigmaNode('INPUT', radioNode);
  }

  if (inputType == 'button' || inputType == 'submit' || inputType == 'reset') {
    let x = Math.round(rect.left);
    let y = Math.round(rect.top);
    let width = Math.round(rect.width);
    let height = Math.round(rect.height);

    const fills: SolidPaint[] = [];
    let rgb = getFigmaRGB(computedStyles.backgroundColor);

    if (rgb) {
      fills.push({
        type: 'SOLID',
        color: {
          r: rgb.r,
          g: rgb.g,
          b: rgb.b,
        },
        blendMode: 'NORMAL',
        visible: true,
        opacity: rgb.a || 1,
        flexDirection: computedStyles.flexDirection || 'none',
      } as SolidPaint);
    }

    const mapTextAlignHorizontal = (
      cssAlign: string | undefined,
    ): 'LEFT' | 'RIGHT' | 'CENTER' | 'JUSTIFIED' | undefined =>
      (
        ({
          left: 'LEFT',
          right: 'RIGHT',
          center: 'CENTER',
          justify: 'JUSTIFIED',
        }) as const
      )[cssAlign as 'left' | 'right' | 'center' | 'justify'];

    const mapTextAlignVertical = (cssAlign: string | undefined): 'TOP' | 'CENTER' | 'BOTTOM' | undefined =>
      (({ top: 'TOP', middle: 'CENTER', bottom: 'BOTTOM' }) as const)[cssAlign as 'top' | 'middle' | 'bottom'];

    const textNode: Partial<TextNode> = {
      type: 'TEXT',
      characters: (element as HTMLInputElement).value?.trim(),
      x: x,
      y: y,
      width: width,
      height: height,
      textAlignHorizontal: mapTextAlignHorizontal(computedStyles.textAlign),
      textAlignVertical: mapTextAlignVertical(computedStyles.verticalAlign),
      fontSize: parseFloat(computedStyles.fontSize),
      fontName: {
        family: computedStyles.fontFamily.replace(/['"]/g, ''),
        style: computedStyles.fontStyle,
      },
      fills: fills,
    };

    const buttonNode: Partial<RectangleNode> = {
      type: 'RECTANGLE',
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      fills: fills,
      topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
      topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
      bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
      bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
      effects: shadow ? [shadow] : [],
    };
    const inputFigmaNode = createFigmaNode('BUTTON', buttonNode);
    const textFigmaNode = createFigmaNode('TXT', textNode);
    inputFigmaNode.children.push(textFigmaNode);

    return inputFigmaNode;
  }

  if (
    inputType == 'email' ||
    inputType == 'text' ||
    inputType == 'search' ||
    inputType == 'list' ||
    inputType == 'number'
  ) {
    let x = Math.round(rect.left);
    let y = Math.round(rect.top);
    let width = Math.round(rect.width);
    let height = Math.round(rect.height);

    const fills: SolidPaint[] = [];
    let rgb = getFigmaRGB(computedStyles.backgroundColor);

    if (rgb) {
      fills.push({
        type: 'SOLID',
        color: {
          r: rgb.r,
          g: rgb.g,
          b: rgb.b,
        },
        blendMode: 'NORMAL',
        visible: true,
        opacity: rgb.a || 1,
        flexDirection: computedStyles.flexDirection || 'none',
      } as SolidPaint);
    }

    const fillsText: SolidPaint[] = [];

    const defaultPlaceholderColor = getFigmaRGB('rgba(178, 178, 178, 1)');

    let rgbText = defaultPlaceholderColor;

    if (rgbText) {
      fillsText.push({
        type: 'SOLID',
        color: {
          r: rgbText.r,
          g: rgbText.g,
          b: rgbText.b,
        },
        blendMode: 'NORMAL',
        visible: true,
        opacity: rgbText.a || 1,
      } as SolidPaint);
    }

    const mapTextAlignHorizontal = (
      cssAlign: string | undefined,
    ): 'LEFT' | 'RIGHT' | 'CENTER' | 'JUSTIFIED' | undefined =>
      (
        ({
          left: 'LEFT',
          right: 'RIGHT',
          center: 'CENTER',
          justify: 'JUSTIFIED',
        }) as const
      )[cssAlign as 'left' | 'right' | 'center' | 'justify'];

    const mapTextAlignVertical = (cssAlign: string | undefined): 'TOP' | 'CENTER' | 'BOTTOM' | undefined =>
      (({ top: 'TOP', middle: 'CENTER', bottom: 'BOTTOM' }) as const)[cssAlign as 'top' | 'middle' | 'bottom'];

    const textFieldNode: Partial<RectangleNode> = {
      type: 'RECTANGLE',
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      fills: fills,
      topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
      topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
      bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
      bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    };
    const inputFigmaNode = createFigmaNode('INPUT', textFieldNode);

    if ((element as HTMLInputElement).placeholder?.trim()) {
      const textNode: Partial<TextNode> = {
        type: 'TEXT',
        characters: (element as HTMLInputElement).placeholder?.trim(),
        x: x,
        y: y,
        width: width,
        height: height,
        textAlignHorizontal: mapTextAlignHorizontal(computedStyles.textAlign),
        textAlignVertical: mapTextAlignVertical(computedStyles.verticalAlign),
        fontSize: parseFloat(computedStyles.fontSize),
        fontName: {
          family: computedStyles.fontFamily.replace(/['"]/g, ''),
          style: computedStyles.fontStyle,
        },
        fills: fillsText,
      };
      const textFigmaNode = createFigmaNode('TXT', textNode);
      inputFigmaNode.children.push(textFigmaNode);
    }

    return inputFigmaNode;
  }

  const inputNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    effects: shadow ? [shadow] : [],
  };

  return createFigmaNode('INPUT', inputNode);
}


/**
 * Converts HTML button or form elements to a Figma RectangleNode by extracting
 * positioning, background fills, border radius, borders/strokes, and visual effects
 */
export function handleButtonFormNode(element: Element): Partial<RectangleNode> {
  const el = element as HTMLButtonElement | HTMLFormElement;
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(el);

  const fills: SolidPaint[] = [];
  let rgb = getFigmaRGB(computedStyles.backgroundColor);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: {
        r: rgb.r,
        g: rgb.g,
        b: rgb.b,
      },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  const ButtonFormNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    name: computedStyles.boxShadow,
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
    strokes: (borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[],
    strokeWeight: borderData?.strokeWeight || 0,
    dashPattern: borderData?.dashPattern || [],
    effects: shadow ? [shadow] : [],
  };

  return ButtonFormNode;
}


/**
 * Converts generic HTML elements (div, span, etc.) to either a Figma GroupNode or RectangleNode
 * based on whether they have border radius, box shadows, or borders - returns RectangleNode
 * if styled, GroupNode if not
 */
export function handleDefaultNode(element: Element): Partial<GroupNode> | Partial<RectangleNode> {
  const el = element as HTMLDivElement | HTMLSpanElement;
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(el);

  const fills: SolidPaint[] = [];
  const rgb = getFigmaRGB(computedStyles.backgroundColor);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: { r: rgb.r, g: rgb.g, b: rgb.b },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  // Extract strokes (borders)
  const borderData = getBorder(computedStyles);

  // Detect border radius or shadow
  const hasBorderRadius =
    parseFloat(computedStyles.borderTopLeftRadius) > 0 ||
    parseFloat(computedStyles.borderTopRightRadius) > 0 ||
    parseFloat(computedStyles.borderBottomLeftRadius) > 0 ||
    parseFloat(computedStyles.borderBottomRightRadius) > 0;

  const hasBoxShadow = computedStyles.boxShadow !== 'none';

  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  if (hasBorderRadius || hasBoxShadow || borderData) {
    return {
      type: 'RECTANGLE',
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      fills,
      topLeftRadius: parseFloat(computedStyles.borderTopLeftRadius) || 0,
      topRightRadius: parseFloat(computedStyles.borderTopRightRadius) || 0,
      bottomLeftRadius: parseFloat(computedStyles.borderBottomLeftRadius) || 0,
      bottomRightRadius: parseFloat(computedStyles.borderBottomRightRadius) || 0,
      strokes: (borderData?.strokes || []).filter((stroke) => stroke !== null) as Paint[],
      strokeWeight: borderData?.strokeWeight || 0,
      dashPattern: borderData?.dashPattern || [],
      effects: shadow ? [shadow] : [],
    };
  }

  return {
    type: 'GROUP',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    backgrounds: fills,
    effects: shadow ? [shadow] : [],
  };
}


/**
 * Converts an HTML anchor (link) element to a Figma LinkUnfurlNode by extracting
 * the href URL and positioning information for link preview functionality
 */
export function handleLinkNode(element: Element): Partial<LinkUnfurlNode> {
  const el = element as HTMLAnchorElement;
  const rect = el.getBoundingClientRect();

  const linkNode: Partial<LinkUnfurlNode> = {
    type: 'LINK_UNFURL',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    linkUnfurlData: { url: element.getAttribute('href') as string, title: null, description: null, provider: null },
  };

  return linkNode;
}


/**
 * Converts an HTML body element to a Figma FrameNode by extracting positioning,
 * background fills, and border radius properties to serve as the document container
 */
export function handleBodyNode(element: Element): Partial<FrameNode> {
  const el = element as HTMLBodyElement;
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(el);

  const fills: SolidPaint[] = [];
  let rgb = getFigmaRGB(computedStyles.backgroundColor);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: {
        r: rgb.r,
        g: rgb.g,
        b: rgb.b,
      },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

  const BodyNode: Partial<FrameNode> = {
    type: 'FRAME',
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fills: fills,
    topLeftRadius: parse(computedStyles.borderTopLeftRadius, rect.height),
    topRightRadius: parse(computedStyles.borderTopRightRadius, rect.height),
    bottomLeftRadius: parse(computedStyles.borderBottomLeftRadius, rect.height),
    bottomRightRadius: parse(computedStyles.borderBottomRightRadius, rect.height),
  };

  return BodyNode;
}


/**
 * Converts an HTML select dropdown element to a Figma node structure containing:
 * - Main RectangleNode for the select box
 * - Child TextNode for the first option text
 * - Child VectorNode for the dropdown arrow icon
 */
export function handleSelectNode(element: Element): FigmaNode {
  const selectEl = element as HTMLSelectElement;
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);

  const fills: SolidPaint[] = [];
  const shadow: DropShadowEffect | null = parseBoxShadow(computedStyles.boxShadow);

  let x = Math.round(rect.left);
  let y = Math.round(rect.top);
  let width = Math.round(rect.width);
  let height = Math.round(rect.height);

  let rgb = getFigmaRGB(computedStyles.color);

  if (rgb) {
    fills.push({
      type: 'SOLID',
      color: { r: rgb.r, g: rgb.g, b: rgb.b },
      blendMode: 'NORMAL',
      visible: true,
      opacity: rgb.a || 1,
      flexDirection: computedStyles.flexDirection || 'none',
    } as SolidPaint);
  }

  // Create select box
  const selectNode: Partial<RectangleNode> = {
    type: 'RECTANGLE',
    x: x,
    y: y,
    width: width,
    height: height,
    fills: fills,
    effects: shadow ? [shadow] : [],
  };
  const figmaSelectNode = createFigmaNode('SELECT', selectNode);

  // Get first option text
  const firstOptionEl = selectEl.options[0];

  if (firstOptionEl) {
    const firstOptionText = firstOptionEl.textContent?.trim() || '';
    const optionStyles = getComputedStyle(firstOptionEl);

    const optionFills: SolidPaint[] = [];

    const optionRGB = getFigmaRGB(optionStyles.color);
    if (optionRGB) {
      optionFills.push({
        type: 'SOLID',
        color: {
          r: optionRGB.r,
          g: optionRGB.g,
          b: optionRGB.b,
        },
        blendMode: 'NORMAL',
        visible: true,
        opacity: optionRGB.a || 1,
      } as SolidPaint);
    }

    const textNode: Partial<TextNode> = {
      type: 'TEXT',
      characters: firstOptionText,
      x: x + 10, // Left padding
      y: y + height / 4, // Centered vertically
      width: width - 30, // Leave space for arrow
      height: height / 2,
      fontSize: parseFloat(optionStyles.fontSize),
      fontName: {
        family: optionStyles.fontFamily.replace(/['"]/g, ''),
        style: optionStyles.fontStyle,
      },
      fills: optionFills, // Apply option-specific fills
    };

    const figmaTextNode = createFigmaNode('TXT', textNode);
    figmaSelectNode.children.push(figmaTextNode);
  }

  // Create dropdown arrow
  const arrowNode: Partial<VectorNode> = {
    type: 'VECTOR',
    x: x + width - 20, // Right side padding
    y: y + height / 2 - 3, // Centered vertically
    fills: fills,
    vectorPaths: [
      {
        windingRule: 'NONZERO',
        data: 'M 0 0 L 6 6 L 12 0', // Simple downward chevron
      },
    ],
  };
  const figmaArrowNode = createFigmaNode('VECTOR', arrowNode);
  figmaSelectNode.children.push(figmaArrowNode);

  return figmaSelectNode;
}
