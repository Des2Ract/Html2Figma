export interface SvgNode extends DefaultShapeMixin, ConstraintMixin {
  type: "SVG";
  svg: string;
}

export type LayerNode = Partial<
  | Partial<RectangleNode>
  | Partial<TextNode>
  | Partial<FrameNode>
  | Partial<SvgNode>
  | Partial<GroupNode>
  | Partial<ComponentNode>
>;

export interface FigmaNode {
  tag: string;
  node: LayerNode;
  children: FigmaNode[];
}

export function createFigmaNode(tag: string, node: LayerNode): FigmaNode {
  return {
    tag,
    node,
    children: [],
  };
}
