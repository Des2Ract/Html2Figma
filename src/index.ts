import { crawlWebPage } from "./crawler.js";
import { parse } from "./parser.js";

async function main() {
  const urls = await crawlWebPage(10);

  const link = "http://127.0.0.1:5500/index.html";

  const figmaTree = await parse(link);

  console.log("Generated Figma-like JSON:");
  console.log(JSON.stringify(figmaTree, null, 2));
}

main();
