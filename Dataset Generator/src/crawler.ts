import puppeteer, { Browser } from 'puppeteer';
import fs from 'fs/promises';
import path from 'path';
import robotsParser from 'robots-parser';
import { URL } from 'url';

const CACHE_FILE: string = path.resolve('./', 'crawledUrls.json');
const SEED_FILE: string = path.resolve('./', 'seedUrls.json');
const MAX_CONCURRENT_PAGES: number = 3;
const NAVIGATION_TIMEOUT: number = 30000; // 30 seconds

type RobotsTxt = ReturnType<typeof robotsParser> | null;
type CrawlResult = string[];

/** Load cached URLs from disk */
async function loadCache(): Promise<Set<string>> {
  try {
    const data = await fs.readFile(CACHE_FILE, 'utf8');
    return new Set(JSON.parse(data));
  } catch {
    return new Set();
  }
}

/** Save crawled URLs to disk */
async function saveCache(urls: Set<string>): Promise<void> {
  await fs.writeFile(CACHE_FILE, JSON.stringify([...urls], null, 2));
}

/** Load seed URLs from disk */
async function loadSeedUrls(): Promise<string[]> {
  try {
    const data = await fs.readFile(SEED_FILE, 'utf8');
    return JSON.parse(data);
  } catch {
    throw new Error('Seed URLs file not found.');
  }
}

/** Fetch and parse robots.txt for a given domain */
async function fetchRobotsTxt(url: string): Promise<RobotsTxt> {
  try {
    const robotsUrl: string = new URL('/robots.txt', url).href;
    const res = await fetch(robotsUrl);
    if (!res.ok) return null;
    return robotsParser(robotsUrl, await res.text());
  } catch {
    return null;
  }
}

/** Normalize URLs by removing fragments and (optionally) trailing slashes */
function normalizeUrl(url: string): string {
  try {
    const u = new URL(url);
    u.hash = ''; // Remove fragment
    if (u.pathname.endsWith('/') && u.pathname !== '/') {
      u.pathname = u.pathname.slice(0, -1);
    }
    return u.toString();
  } catch {
    return url;
  }
}

/** Crawl a single page with a retry mechanism and return all discovered (normalized) links */
async function crawlPage(url: string, browser: Browser, robotsRules: Map<string, RobotsTxt>): Promise<string[]> {
  const MAX_RETRIES = 3;
  let attempt = 0;
  let links: string[] = [];
  console.log(`Starting crawl for: ${url}`);
  while (attempt < MAX_RETRIES) {
    attempt++;
    let page;
    try {
      page = await browser.newPage();
      await page.setUserAgent(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      );
      const baseUrl: string = new URL(url).origin;
      if (!robotsRules.has(baseUrl)) {
        console.log(`Fetching robots.txt for ${baseUrl}`);
        const robots = await fetchRobotsTxt(baseUrl);
        robotsRules.set(baseUrl, robots);
      }
      const robots: RobotsTxt = robotsRules.get(baseUrl) || null;
      if (robots && !robots.isAllowed(url)) {
        console.warn(`Blocked by robots.txt: ${url}`);
        return [];
      }
      console.log(`Attempt ${attempt}: Navigating to ${url}`);
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAVIGATION_TIMEOUT });
      console.log(`Extracting links from ${url}`);
      links = await page.evaluate(() => {
        try {
          return Array.from(document.querySelectorAll('a'))
            .map((a) => a.href)
            .filter((href) => href.startsWith('http'));
        } catch (e) {
          return [];
        }
      });
      console.log(`Found ${links.length} links on ${url}`);
      await page.close();
      break; // If successful, exit the loop
    } catch (err) {
      console.warn(`Attempt ${attempt} failed for ${url}:`, err);
      if (attempt === MAX_RETRIES) {
        console.error(`Failed to crawl ${url} after ${MAX_RETRIES} attempts`);
      } else {
        // Wait for 1 second before retrying
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      // Ensure page is closed if an error occurs
      if (page) {
        try {
          await page.close();
        } catch (_) {}
      }
    }
  }
  console.log(`Finished crawl for: ${url}`);
  return links.map(normalizeUrl);
}

/** Crawl a level of URLs with a concurrency limit */
async function crawlLevel(urls: string[], browser: Browser, robotsRules: Map<string, RobotsTxt>): Promise<string[]> {
  console.log(`Processing a chunk of ${urls.length} URLs...`);
  const results: string[] = [];
  // Process URLs in chunks to respect the MAX_CONCURRENT_PAGES limit
  for (let i = 0; i < urls.length; i += MAX_CONCURRENT_PAGES) {
    const chunk = urls.slice(i, i + MAX_CONCURRENT_PAGES);
    console.log(`Processing chunk: ${chunk.join(', ')}`);
    const chunkResults = await Promise.all(chunk.map((url) => crawlPage(url, browser, robotsRules)));
    for (const res of chunkResults) {
      results.push(...res);
    }
  }
  console.log(`Level complete. Found a total of ${results.length} links in this level.`);
  return results;
}

/** Main BFS crawl function */
export async function crawlWebPage(crawlSize: number, useCache: boolean): Promise<CrawlResult> {
  const cachedUrls = await loadCache();
  if (useCache && cachedUrls.size > 0) {
    console.log('Loaded URLs from cache.');
    return [...cachedUrls];
  }

  const seedUrlsRaw = await loadSeedUrls();
  // Normalize seed URLs and remove duplicates
  const seedUrls = Array.from(new Set(seedUrlsRaw.map(normalizeUrl)));

  const browser = await puppeteer.launch({ headless: true });
  const robotsRules: Map<string, RobotsTxt> = new Map();

  // All visited URLs will be stored here.
  const visited = new Set<string>(seedUrls);
  let currentLevel: string[] = seedUrls;
  let level = 0;

  while (currentLevel.length > 0 && visited.size < crawlSize) {
    level++;
    console.log(
      `\n=== Crawling level ${level} with ${currentLevel.length} URLs. Total visited so far: ${visited.size} ===`,
    );
    const nextLevelLinks = await crawlLevel(currentLevel, browser, robotsRules);

    // Filter out already visited links and eliminate duplicates
    const nextLevelSet = new Set<string>();
    for (const link of nextLevelLinks) {
      const normalized = normalizeUrl(link);
      if (normalized && !visited.has(normalized)) {
        nextLevelSet.add(normalized);
      }
    }

    if (nextLevelSet.size === 0) {
      console.log(`No new links found at level ${level}. Stopping crawl.`);
      break;
    }

    // Add new links to visited (up to crawlSize)
    for (const link of nextLevelSet) {
      if (visited.size < crawlSize) {
        visited.add(link);
      } else {
        break;
      }
    }

    currentLevel = Array.from(nextLevelSet);
  }

  await browser.close();
  await saveCache(visited);
  console.log('Crawl complete. URLs cached.');
  return Array.from(visited);
}
