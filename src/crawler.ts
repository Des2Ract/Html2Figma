import fetch from "node-fetch";
import { JSDOM } from "jsdom";

export async function crawlWebPage(url: string) {
  try {
    console.log(`Crawling: ${url}`);

    // Fetch the HTML
    const response = await fetch(url);
    const html = await response.text();

    // Parse HTML with JSDOM
    const dom = new JSDOM(html);
    const document = dom.window.document;

    // Extract all external CSS links
    const cssLinks = Array.from(
      document.querySelectorAll('link[rel="stylesheet"]')
    )
      .map((link) => link.href)
      .filter((href) => href.startsWith("http") || href.startsWith("//"));

    // Fetch and store CSS content
    const cssContents = await Promise.all(
      cssLinks.map(async (cssUrl) => {
        try {
          const cssResponse = await fetch(cssUrl);
          return await cssResponse.text();
        } catch {
          console.warn(`Failed to fetch CSS: ${cssUrl}`);
          return "";
        }
      })
    );

    return {
      html,
      cssContents,
    };
  } catch (error) {
    console.error(`Error crawling webpage: ${error}`);
    return null;
  }
}
