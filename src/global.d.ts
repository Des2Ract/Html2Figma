export {};

declare global {
  interface Window {
    createFigmaNode: typeof import("./figma_node").createFigmaNode;
    extractFigmaNode: typeof import("./FigmaNodeExtractor").extractFigmaNode;
    getFigmaRGB: typeof import("./utils").getFigmaRGB;
    handleTextNode: typeof import("./FigmaComponentHandlers").handleTextNode;
    handleSvgNode: typeof import("./FigmaComponentHandlers").handleSvgNode;
    handleImageNode: typeof import("./FigmaComponentHandlers").handleImageNode;
  }
}
