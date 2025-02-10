import puppeteer from "puppeteer";
import { FigmaNode } from "./figma_node.js";
import { parseHTMLToFigmaNode } from "./html_parser.js";
import { createFigmaNode } from "./figma_node.js";
import { extractFigmaNode, handleTextNode } from "./FigmaNodeExtractor.js";
import { getFigmaRGB } from "./utils.js";

export async function parse(url: string): Promise<FigmaNode> {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "networkidle2" });

  // Bundle necessary logic into an object
  const logicBundle = {
    parseHTMLToFigmaNode: parseHTMLToFigmaNode.toString(),
    createFigmaNode: createFigmaNode.toString(),
    extractFigmaNode: extractFigmaNode.toString(),
    getFigmaRGB: getFigmaRGB.toString(),
    handleTextNode: handleTextNode.toString(),
  };

  const result = await page.evaluate((logic) => {
    // Reconstruct functions in the browser
    const parseHTMLToFigmaNode = new Function(
      `return ${logic.parseHTMLToFigmaNode}`
    )();
    const createFigmaNode = new Function(`return ${logic.createFigmaNode}`)();
    const extractFigmaNode = new Function(`return ${logic.extractFigmaNode}`)();
    const getFigmaRGB = new Function(`return ${logic.getFigmaRGB}`)();
    const handleTextNode = new Function(`return ${logic.handleTextNode}`)();

    // Use functions to process the document
    window.createFigmaNode = createFigmaNode;
    window.extractFigmaNode = extractFigmaNode;
    window.getFigmaRGB = getFigmaRGB;
    window.handleTextNode = handleTextNode;
    return parseHTMLToFigmaNode(document.body);
  }, logicBundle);

  await browser.close();
  return result;
}
