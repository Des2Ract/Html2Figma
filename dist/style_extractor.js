export function extractStyles(element) {
    var _a, _b;
    const styleObj = {}; // Use styleObj to store the styles
    const computedStyles = ((_b = (_a = element.ownerDocument) === null || _a === void 0 ? void 0 : _a.defaultView) !== null && _b !== void 0 ? _b : window).getComputedStyle(element);
    // Loop through the computed styles
    for (let i = 0; i < computedStyles.length; i++) {
        const property = computedStyles[i]; // Get the property name
        const value = computedStyles.getPropertyValue(property); // Get the property value
        styleObj[property] = value; // Store the property and value in styleObj
    }
    return styleObj; // Return the collected styles
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
