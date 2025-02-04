import { crawlWebPage } from "./crawler";
import { parseHTMLToFigmaNode } from "./html_parser";
import { JSDOM } from "jsdom";

async function main() {
  const url = "https://google.com"; // Replace with your target URL
  const result = await crawlWebPage(url);

  if (result) {
    const dom = new JSDOM(result.html);
    const document = dom.window.document;
    const rootElement = document.body;

    const figmaTree = parseHTMLToFigmaNode(rootElement);
    console.log("Generated Figma-like JSON:");
    console.log(JSON.stringify(figmaTree, null, 2));
  }
}

main();
