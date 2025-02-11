import puppeteer from 'puppeteer';
import fs from 'fs/promises';
import path from 'path';

const CACHE_FILE = path.resolve('./', 'crawledUrls.json');

// Helper to read cache file if available
async function loadCache(): Promise<string[] | null> {
  try {
    const data = await fs.readFile(CACHE_FILE, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    return null; // Cache does not exist
  }
}

// Helper to save crawled URLs to cache
async function saveCache(urls: string[]): Promise<void> {
  await fs.writeFile(CACHE_FILE, JSON.stringify(urls, null, 2));
}

// Main crawl function
export async function crawlWebPage(crawlSize: number): Promise<string[]> {
  // Check if cache exists
  const cachedUrls = await loadCache();
  if (cachedUrls) {
    console.log('Loaded URLs from cache.');
    return cachedUrls;
  }

  console.log('No cache found. Starting crawl...');

  const startUrl = 'https://www.berkshirehathaway.com/'; // Replace with the target start URL
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(startUrl);

  const crawledUrls: Set<string> = new Set();
  crawledUrls.add(startUrl);

  let pagesToCrawl: string[] = [startUrl];

  while (pagesToCrawl.length > 0 && crawledUrls.size < crawlSize) {
    const currentUrl = pagesToCrawl.shift()!;
    try {
      await page.goto(currentUrl, { waitUntil: 'domcontentloaded' });

      const links = await page.$$eval('a', (anchorElements) =>
        anchorElements.map((a) => a.href).filter((href) => href.startsWith('http')),
      );

      links.forEach((link) => {
        if (!crawledUrls.has(link) && crawledUrls.size < crawlSize) {
          crawledUrls.add(link);
          pagesToCrawl.push(link);
        }
      });
    } catch (err) {
      console.warn(`Failed to crawl ${currentUrl}:`, err);
    }
  }

  await browser.close();

  const resultUrls = Array.from(crawledUrls);

  // Save results to cache
  await saveCache(resultUrls);
  console.log('Crawl complete. URLs cached.');

  return resultUrls;
}
