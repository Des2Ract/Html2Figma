interface FigmaColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export function getFigmaRGB(colorString?: string | null): FigmaColor | null {
  if (!colorString) return null;

  const match = colorString.match(
    /rgba?\((\d+), (\d+), (\d+)(?:, ([\d.]+))?\)/
  );
  if (!match) return null;

  const [, r, g, b, a] = match;
  const alpha = a ? parseFloat(a) : 1;

  if (parseFloat(a || "1") === 0) return null;

  return {
    r: parseInt(r, 10) / 255,
    g: parseInt(g, 10) / 255,
    b: parseInt(b, 10) / 255,
    a: alpha,
  };
}
