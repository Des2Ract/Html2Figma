import { promises as fs } from 'fs';
import { crawlWebPage } from './crawler.js';
import { parse } from './parser.js';
import path from 'path';

async function main() {
  try {
    // Read links from links.json

    // const urls = await crawlWebPage(10000, true);
    const urls = ['http://127.0.0.1:5500/test/index.html'];

    // Ensure the data folder exists
    const dataFolder = 'json_data';
    await fs.mkdir(dataFolder, { recursive: true });

    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];
      console.log(`Processing: ${url}`);

      try {
        const figmaTree = await parse(url);
        const json = JSON.stringify(figmaTree, null, 2);

        const filePath = path.join(dataFolder, `figmaTree_${i + 1}.json`);
        console.log(`Generated Figma-like JSON for url ${i + 1}`);
        await fs.writeFile(filePath, json, 'utf-8');
        console.log(`Figma JSON exported to ${filePath}`);
      } catch (error) {
        console.error(`Error processing ${url}, restarting...`, error);
      }
    }
  } catch (error) {
    console.error('Error processing links: ', error);
  }
}

main();
