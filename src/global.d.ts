export {};

declare global {
  interface Window {
    createFigmaNode: typeof import("./figma_node").createFigmaNode;
    extractFigmaNode: typeof import("./FigmaNodeExtractor").extractFigmaNode;
    getFigmaRGB: typeof import("./utils").getFigmaRGB;
    handleTextNode: typeof import("./FigmaNodeExtractor").handleTextNode;
  }
}
