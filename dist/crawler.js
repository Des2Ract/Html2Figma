var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import fetch from "node-fetch";
import { JSDOM } from "jsdom";
export function crawlWebPage(url) {
    return __awaiter(this, void 0, void 0, function* () {
        try {
            console.log(`Crawling: ${url}`);
            // Fetch the HTML
            const response = yield fetch(url);
            const html = yield response.text();
            // Parse HTML with JSDOM
            const dom = new JSDOM(html);
            const document = dom.window.document;
            // Extract all external CSS links
            const cssLinks = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
                .map((link) => link.href)
                .filter((href) => href.startsWith("http") || href.startsWith("//"));
            // Fetch and store CSS content
            const cssContents = yield Promise.all(cssLinks.map((cssUrl) => __awaiter(this, void 0, void 0, function* () {
                try {
                    const cssResponse = yield fetch(cssUrl);
                    return yield cssResponse.text();
                }
                catch (_a) {
                    console.warn(`Failed to fetch CSS: ${cssUrl}`);
                    return "";
                }
            })));
            return {
                html,
                cssContents,
            };
        }
        catch (error) {
            console.error(`Error crawling webpage: ${error}`);
            return null;
        }
    });
}
