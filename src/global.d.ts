export {};

declare global {
  interface Window {
    createFigmaNode: typeof import('./figma_node').createFigmaNode;
    extractFigmaNode: typeof import('./FigmaNodeExtractor').extractFigmaNode;
    getFigmaRGB: typeof import('./utils').getFigmaRGB;
    handleTextNode: typeof import('./FigmaComponentHandlers').handleTextNode;
    handleSvgNode: typeof import('./FigmaComponentHandlers').handleSvgNode;
    handleImageNode: typeof import('./FigmaComponentHandlers').handleImageNode;
    handlePictureNode: typeof import('./FigmaComponentHandlers').handlePictureNode;
    handleVideoNode: typeof import('./FigmaComponentHandlers').handleVideoNode;
    handleLineNode: typeof import('./FigmaComponentHandlers').handleLineNode;
    handleButtonFormNode: typeof import('./FigmaComponentHandlers').handleButtonFormNode;
    handleDivSpanNode: typeof import('./FigmaComponentHandlers').handleDivSpanNode;
    handleLinkNode: typeof import('./FigmaComponentHandlers').handleLinkNode;
    handleBodyNode: typeof import('./FigmaComponentHandlers').handleBodyNode;
    handleInputNode: typeof import('./FigmaComponentHandlers').handleInputNode;

  }
}
