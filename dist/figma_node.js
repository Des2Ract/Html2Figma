export function createFigmaNode(tagName, styles) {
    return {
        type: "node",
        tagName,
        styles,
        children: [],
    };
}
