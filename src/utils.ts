// Utility function to handle deep cloning of an object
export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

// Utility to handle merging styles
export function mergeStyles(
  styles1: Record<string, string>,
  styles2: Record<string, string>
): Record<string, string> {
  return { ...styles1, ...styles2 };
}

// Utility function to format styles (for example, convert to camelCase)
export function formatStyles(
  styles: Record<string, string>
): Record<string, string> {
  const formattedStyles: Record<string, string> = {};
  for (const key in styles) {
    if (styles.hasOwnProperty(key)) {
      const formattedKey = key.replace(/-([a-z])/g, (match) =>
        match[1].toUpperCase()
      ); // Convert to camelCase
      formattedStyles[formattedKey] = styles[key];
    }
  }
  return formattedStyles;
}
