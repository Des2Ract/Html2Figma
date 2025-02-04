export function extractStyles(element) {
    const computedStyles = getComputedStyle(element);
    const styleObj = {};
    for (const prop of computedStyles) {
        styleObj[prop] = computedStyles.getPropertyValue(prop);
    }
    return styleObj;
}
export function applyStylesToFigmaNode(element) {
    const tagName = element.tagName.toLowerCase();
    const styles = extractStyles(element);
    return {
        type: "node",
        tagName,
        styles,
        children: [],
    };
}
