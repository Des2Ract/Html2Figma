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
  const borderStyles = new Set<string>();
  let strokeWeight = 0;
  let dashPattern: number[] = [];
  let strokePaint: Paint | null = null;

  for (const dir of directions) {
    const border = computedStyle.getPropertyValue(`border-${dir.toLowerCase()}`);
    if (!border || typeof border !== 'string') continue;

    const parsed = border.match(borderRegex);
    if (!parsed) continue;

    const [, width, type, color] = parsed;
    if (!width || width === '0' || type === 'none' || !color) continue;

    const rgb = getFigmaRGB(color);
    if (!rgb) continue;

    borderStyles.add(`${width} ${type} ${color}`);
    strokeWeight = Math.max(strokeWeight, parseFloat(width));

    if (!strokePaint) {
      strokePaint = {
        type: 'SOLID',
        color: { r: rgb.r, g: rgb.g, b: rgb.b },
        opacity: rgb.a || 1,
      };
    }

    if (type === 'dashed') {
      dashPattern = [6, 4];
    } else if (type === 'dotted') {
      dashPattern = [2, 2];
    }
  }

  if (borderStyles.size === 1 && strokePaint) {
    return { strokes: [strokePaint], strokeWeight, dashPattern };
  } else if (borderStyles.size > 1) {
    // Different borders on different sides (e.g., mixed styles) -> Keep them separate
    return { strokes: [strokePaint], strokeWeight, dashPattern, individualSides: true };
  }

  return null;
}

export function parseBoxShadow(cssShadow: string | null): DropShadowEffect | null {
  if (!cssShadow || cssShadow === 'none') return null;

  // Updated regex to handle color appearing first OR last
  const shadowRegex =
    /(rgba?\([^)]+\)|#[0-9a-fA-F]+)?\s*(-?\d+\.?\d*)px\s*(-?\d+\.?\d*)px\s*(\d+\.?\d*)px\s*(\d+\.?\d*)?px?\s*(rgba?\([^)]+\)|#[0-9a-fA-F]+)?/;
  const match = cssShadow.match(shadowRegex);

  if (!match) return null;

  // Identify where the color is
  const colorString = match[1] || match[6]; // Color can be at the start or end

  // Extract numerical values
  const offsetX = parseFloat(match[2]);
  const offsetY = parseFloat(match[3]);
  const blurRadius = parseFloat(match[4]);
  const spreadRadius = match[5] ? parseFloat(match[5]) : 0; // Default to 0 if not present

  // Convert color to RGBA format
  let parsedColor = { r: 0, g: 0, b: 0, a: 1 }; // Default black
  if (colorString) {
    const rgbaMatch = colorString.match(/rgba?\((\d+), (\d+), (\d+),? (\d?.?\d+)?\)/);
    if (rgbaMatch) {
      parsedColor = {
        r: parseInt(rgbaMatch[1], 10) / 255,
        g: parseInt(rgbaMatch[2], 10) / 255,
        b: parseInt(rgbaMatch[3], 10) / 255,
        a: rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1,
      };
    }
  }

  return {
    type: 'DROP_SHADOW',
    color: parsedColor,
    offset: { x: offsetX, y: offsetY },
    radius: blurRadius,
    visible: true,
    blendMode: 'NORMAL',
  };
}
