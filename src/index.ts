import { promises as fs } from "fs";
import { crawlWebPage } from "./crawler.js";
import { parse } from "./parser.js";

async function main() {
  const urls = await crawlWebPage(10);

  const link = "http://127.0.0.1:5500/test/index.html";

  const figmaTree = await parse(link);
  const json = JSON.stringify(figmaTree, null, 2);

  console.log("Generated Figma-like JSON:");
  console.log(json);

  await fs.writeFile("figmaTree.json", json, "utf-8");
  console.log("Figma JSON exported to figmaTree.json");
}

main();
