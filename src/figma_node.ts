export interface SvgNode extends DefaultShapeMixin, ConstraintMixin {
  type: "SVG";
  svg: string;
}

export type LayerNode = Partial<
  RectangleNode | TextNode | FrameNode | SvgNode | GroupNode | ComponentNode
>;

export interface FigmaNode {
  tag: string;
  node: LayerNode;
  children: FigmaNode[];
}
