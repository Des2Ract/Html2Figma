import { promises as fs } from 'fs';
import { crawlWebPage } from './crawler.js';
import { parse } from './parser.js';
import path from 'path';

async function processLinks() {
  try {
    // Read links from links.json
    const linksData = await fs.readFile('links.json', 'utf-8');
    const links = JSON.parse(linksData);

    // Ensure the data folder exists
    const dataFolder = 'data';
    await fs.mkdir(dataFolder, { recursive: true });

    for (let i = 0; i < links.length; i++) {
      const link = links[i];
      console.log(`Processing: ${link}`);

      try {
        const figmaTree = await parse(link);
        const json = JSON.stringify(figmaTree, null, 2);

        const filePath = path.join(dataFolder, `figmaTree_${i + 1}.json`);
        console.log(`Generated Figma-like JSON for link ${i + 1}`);
        await fs.writeFile(filePath, json, 'utf-8');
        console.log(`Figma JSON exported to ${filePath}`);
      } catch (error) {
        console.error(`Error processing ${link}, restarting...`, error);
      }
    }
  } catch (error) {
    console.error('Error processing links: ', error);
  }
}

processLinks();
