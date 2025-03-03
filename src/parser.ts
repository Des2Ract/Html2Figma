import puppeteer from 'puppeteer';
import { FigmaNode } from './figma_node.js';
import { parseHTMLToFigmaNode } from './html_parser.js';
import { createFigmaNode } from './figma_node.js';
import { extractFigmaNode } from './FigmaNodeExtractor.js';
import { getBorder, getFigmaRGB, parseBoxShadow } from './utils.js';
import {
  handleButtonFormNode,
  handleDefaultNode,
  handleImageNode,
  handleLineNode,
  handleLinkNode,
  handlePictureNode,
  handleSvgNode,
  handleTextNode,
  handleVideoNode,
  handleBodyNode,
  handleInputNode,
  handleSelectNode,
} from './FigmaComponentHandlers.js';

export async function parse(url: string): Promise<FigmaNode> {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', // Path to Edge
    headless: true,
    defaultViewport: null,
    args: ['--start-maximized'],
  });

  const page = await browser.newPage();

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 100000 });

  // Scroll to the bottom of the page
  await page.evaluate(async () => {
    let totalHeight = 0;
    const distance = 20;
    const startTime = Date.now(); // Get the start time

    while (Date.now() - startTime < 5000) {
      // Stop after 10ms
      window.scrollBy(0, distance);
      totalHeight += distance;
      await new Promise((resolve) => setTimeout(resolve, 1)); // Small delay to allow scrolling
    }
  });

  // // Function to click buttons for cookies or pop-ups
  // const clickPopupButtons = async () => {
  //   const buttonTexts = [
  //     'Accept',
  //     'Continue',
  //     'Agree',
  //     'Yes, I accept',
  //     'I agree',
  //     'Got it',
  //     'Allow all',
  //     'Accept all',
  //   ];

  //   await page.evaluate((buttonTexts) => {
  //     document.querySelectorAll('button, div, a').forEach((btn) => {
  //       const element = btn as HTMLElement; // Cast Element to HTMLElement
  //       if (
  //         element.innerText &&
  //         buttonTexts.some((text) => element.innerText.trim().toLowerCase().includes(text.toLowerCase()))
  //       ) {
  //         element.click();
  //       }
  //     });
  //   }, buttonTexts);
  // };

  // await new Promise((resolve) => setTimeout(resolve, 2000));
  // await clickPopupButtons();

  await page.evaluate(async () => {
    const distance = 100; // Number of pixels to scroll up each step
    const startTime = Date.now(); // Get the start time

    while (Date.now() - startTime < 1000) {
      // Stop after 1 second
      window.scrollBy(0, -distance); // Scroll up
      await new Promise((resolve) => setTimeout(resolve, 1)); // Small delay to allow scrolling
    }
  });
  // Wait for 10 seconds (10,000 milliseconds)
  await new Promise((resolve) => setTimeout(resolve, 10000));

  // Now you can continue with further actions

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
    handleDefaultNode: handleDefaultNode.toString(),
    handleLinkNode: handleLinkNode.toString(),
    handleBodyNode: handleBodyNode.toString(),
    handleInputNode: handleInputNode.toString(),
    handleSelectNode: handleSelectNode.toString(),

    getBorder: getBorder.toString(),
    parseBoxShadow: parseBoxShadow.toString(),
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
    const handleDefaultNode = new Function(`return ${logic.handleDefaultNode}`)();
    const handleLinkNode = new Function(`return ${logic.handleLinkNode}`)();
    const handleBodyNode = new Function(`return ${logic.handleBodyNode}`)();
    const handleInputNode = new Function(`return ${logic.handleInputNode}`)();
    const handleSelectNode = new Function(`return ${logic.handleSelectNode}`)();
    const getBorder = new Function(`return ${logic.getBorder}`)();
    const parseBoxShadow = new Function(`return ${logic.parseBoxShadow}`)();

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
    window.handleDefaultNode = handleDefaultNode;
    window.handleLinkNode = handleLinkNode;
    window.handleBodyNode = handleBodyNode;
    window.handleInputNode = handleInputNode;
    window.handleSelectNode = handleSelectNode;

    window.getBorder = getBorder;
    window.parseBoxShadow = parseBoxShadow;

    return parseHTMLToFigmaNode(document.body);
  }, logicBundle);

  await browser.close();
  return result;
}
