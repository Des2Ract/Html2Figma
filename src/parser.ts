import puppeteer from 'puppeteer';
import { FigmaNode } from './figma_node.js';
import { parseHTMLToFigmaNode } from './html_parser.js';
import { createFigmaNode } from './figma_node.js';
import { extractFigmaNode } from './FigmaNodeExtractor.js';
import { getFigmaRGB } from './utils.js';
import {
  handleButtonFormNode,
  handleDivSpanNode,
  handleImageNode,
  handleLineNode,
  handleLinkNode,
  handlePictureNode,
  handleSvgNode,
  handleTextNode,
  handleVideoNode,
  handleBodyNode,
  handleInputNode,
  handleGroupNode,
} from './FigmaComponentHandlers.js';

export async function parse(url: string): Promise<FigmaNode> {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle2' });

  // Bundle necessary logic into an object
  const logicBundle = {
    parseHTMLToFigmaNode: parseHTMLToFigmaNode.toString(),
    createFigmaNode: createFigmaNode.toString(),
    extractFigmaNode: extractFigmaNode.toString(),
    getFigmaRGB: getFigmaRGB.toString(),
    handleTextNode: handleTextNode.toString(),
    handleSvgNode: handleSvgNode.toString(),
    handleImageNode: handleImageNode.toString(),
    handlePictureNode: handlePictureNode.toString(),
    handleVideoNode: handleVideoNode.toString(),
    handleLineNode: handleLineNode.toString(),
    handleButtonFormNode: handleButtonFormNode.toString(),
    handleDivSpanNode: handleDivSpanNode.toString(),
    handleLinkNode: handleLinkNode.toString(),
    handleBodyNode: handleBodyNode.toString(),
    handleInputNode: handleInputNode.toString(),
    handleGroupNode: handleGroupNode.toString(),
  };

  const result = await page.evaluate((logic) => {
    // Reconstruct functions in the browser
    const parseHTMLToFigmaNode = new Function(`return ${logic.parseHTMLToFigmaNode}`)();
    const createFigmaNode = new Function(`return ${logic.createFigmaNode}`)();
    const extractFigmaNode = new Function(`return ${logic.extractFigmaNode}`)();
    const getFigmaRGB = new Function(`return ${logic.getFigmaRGB}`)();
    const handleTextNode = new Function(`return ${logic.handleTextNode}`)();
    const handleSvgNode = new Function(`return ${logic.handleSvgNode}`)();
    const handleImageNode = new Function(`return ${logic.handleImageNode}`)();
    const handlePictureNode = new Function(`return ${logic.handlePictureNode}`)();
    const handleVideoNode = new Function(`return ${logic.handleVideoNode}`)();
    const handleLineNode = new Function(`return ${logic.handleLineNode}`)();
    const handleButtonFormNode = new Function(`return ${logic.handleButtonFormNode}`)();
    const handleDivSpanNode = new Function(`return ${logic.handleDivSpanNode}`)();
    const handleLinkNode = new Function(`return ${logic.handleLinkNode}`)();
    const handleBodyNode = new Function(`return ${logic.handleBodyNode}`)();
    const handleInputNode = new Function(`return ${logic.handleInputNode}`)();
    const handleGroupNode = new Function(`return ${logic.handleGroupNode}`)();

    // Use functions to process the document
    window.createFigmaNode = createFigmaNode;
    window.extractFigmaNode = extractFigmaNode;
    window.getFigmaRGB = getFigmaRGB;
    window.handleTextNode = handleTextNode;
    window.handleSvgNode = handleSvgNode;
    window.handleImageNode = handleImageNode;
    window.handlePictureNode = handlePictureNode;
    window.handleVideoNode = handleVideoNode;
    window.handleLineNode = handleLineNode;
    window.handleButtonFormNode = handleButtonFormNode;
    window.handleDivSpanNode = handleDivSpanNode;
    window.handleLinkNode = handleLinkNode;
    window.handleBodyNode = handleBodyNode;
    window.handleInputNode = handleInputNode;
    window.handleGroupNode = handleGroupNode;

    return parseHTMLToFigmaNode(document.body);
  }, logicBundle);

  await browser.close();
  return result;
}
