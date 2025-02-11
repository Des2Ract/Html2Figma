import { LayerNode } from './figma_node.js';
import { getFigmaRGB } from './utils.js';

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
  };
  return textnode;
}

export function handleSvgNode(element: Element): Partial<LayerNode> {
  const rect = element.getBoundingClientRect();

  const svgNode: Partial<LayerNode> = {
    type: 'SVG',
    svg: element.outerHTML,
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
  };

  return svgNode;
}

export function handleImageNode(element: Element): Partial<RectangleNode> {
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);
  const url = (element as HTMLImageElement).src as string;

  const fills = [
    {
      url: url as string,
      type: 'IMAGE',
      scaleMode: computedStyles.objectFit === 'contain' ? 'FIT' : 'FILL',
      imageHash: null,
    } as ImagePaint,
  ] as ImagePaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

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
  };

  return imageNode;
}

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
    } as ImagePaint,
  ] as ImagePaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

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
  };

  return pictureNode;
}

export function handleVideoNode(element: Element): Partial<RectangleNode> {
  const rect = element.getBoundingClientRect();
  const computedStyles = getComputedStyle(element);
  const url = (element as HTMLVideoElement).poster as string;

  const fills = [
    {
      url: url,
      type: 'IMAGE',
      scaleMode: computedStyles.objectFit === 'contain' ? 'FIT' : 'FILL',
      imageHash: null,
    } as ImagePaint,
  ] as ImagePaint[];

  const handlePX = (v: string): number => (/px$/.test(v) || v === '0' ? parseFloat(v) : 0);
  const handlePercent = (v: string): number => (/^(\d+)%$/.test(v) ? parseInt(v) / 100 : 0);
  const parse = (borderRadius: string, height: number): number =>
    handlePX(borderRadius) ? handlePX(borderRadius) : handlePercent(borderRadius) * height;

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
  };

  return videoNode;
}
