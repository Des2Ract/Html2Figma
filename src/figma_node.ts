export interface FigmaNode {
  type: string;
  tagName: string;
  styles: Record<string, string>;
  children: FigmaNode[];
}

export function createFigmaNode(
  tagName: string,
  styles: Record<string, string>
): FigmaNode {
  return {
    type: "node",
    tagName,
    styles,
    children: [],
  };
}
