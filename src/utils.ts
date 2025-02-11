interface FigmaColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export function getFigmaRGB(colorString?: string | null): FigmaColor | null {
  if (!colorString) return null;

  const match = colorString.match(/rgba?\((\d+), (\d+), (\d+)(?:, ([\d.]+))?\)/);
  if (!match) return null;

  const [, r, g, b, a] = match;
  const alpha = a ? parseFloat(a) : 1;

  if (parseFloat(a || '1') === 0) return null;

  return {
    r: parseInt(r, 10) / 255,
    g: parseInt(g, 10) / 255,
    b: parseInt(b, 10) / 255,
    a: alpha,
  };
}

export function getBorder(computedStyle: CSSStyleDeclaration) {
  const borderRegex = /^([\d\.]+)px\s*(\w+)\s*(.*)$/;
  const directions = ['Top', 'Left', 'Right', 'Bottom'];
  const strokes: Paint[] = [];
  let strokeWeight = 0;
  let dashPattern: number[] = [];

  for (const dir of directions) {
    const border = computedStyle.getPropertyValue(`border-${dir.toLowerCase()}`);
    if (!border || typeof border !== 'string') continue;

    const parsed = border.match(borderRegex);
    if (!parsed) continue;

    const [, width, type, color] = parsed;
    if (!width || width === '0' || type === 'none' || !color) continue;

    const rgb = getFigmaRGB(color);
    if (!rgb) continue;

    const strokePaint: Paint = {
      type: 'SOLID',
      color: { r: rgb.r, g: rgb.g, b: rgb.b },
      opacity: rgb.a || 1,
    };

    strokeWeight = Math.max(strokeWeight, parseFloat(width));

    if (type === 'dashed') {
      dashPattern = [6, 4];
    } else if (type === 'dotted') {
      dashPattern = [2, 2];
    }

    strokes.push(strokePaint);
  }

  return strokes.length ? { strokes, strokeWeight, dashPattern } : null;
}
