import { crawlWebPage } from "./crawler.js";
import { parseHTMLToFigmaNode } from "./html_parser.js";
import { JSDOM } from "jsdom";
const { window } = new JSDOM("<!doctype html><html><body></body></html>");
const { document } = window;

async function main() {
  const url = "https://www.berkshirehathaway.com/";
  const result = await crawlWebPage(url);

  if (result) {
    try {
      // Initialize JSDOM with the fetched HTML
      const dom = new JSDOM(result.html, {
        pretendToBeVisual: true, // Makes jsdom behave more like a browser (can help with some rendering issues)
      });
      const document = dom.window.document;
      const rootElement = document.body;

      const figmaTree = parseHTMLToFigmaNode(rootElement);
      // TODO: use result.cssContents
      console.log("Generated Figma-like JSON:");
      console.log(JSON.stringify(figmaTree, null, 2));
    } catch (error) {
      console.error("Error parsing HTML with JSDOM:", error);
    }
  } else {
    console.error("Failed to fetch the webpage content.");
  }
}

main();
