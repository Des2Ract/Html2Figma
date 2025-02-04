var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { crawlWebPage } from "./crawler.js";
import { parseHTMLToFigmaNode } from "./html_parser.js";
import { JSDOM } from "jsdom";
const { window } = new JSDOM("<!doctype html><html><body></body></html>");
const { document } = window;
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const url = "https://motherfuckingwebsite.com/";
        const result = yield crawlWebPage(url);
        if (result) {
            try {
                // Initialize JSDOM with the fetched HTML
                const dom = new JSDOM(result.html, {
                    pretendToBeVisual: true, // Makes jsdom behave more like a browser (can help with some rendering issues)
                });
                const document = dom.window.document;
                const rootElement = document.body;
                const figmaTree = parseHTMLToFigmaNode(rootElement);
                console.log("Generated Figma-like JSON:");
                console.log(JSON.stringify(figmaTree, null, 2));
            }
            catch (error) {
                console.error("Error parsing HTML with JSDOM:", error);
            }
        }
        else {
            console.error("Failed to fetch the webpage content.");
        }
    });
}
main();
