var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { crawlWebPage } from "./crawler";
import { parseHTMLToFigmaNode } from "./html_parser";
import { JSDOM } from "jsdom";
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const url = "https://example.com"; // Replace with your target URL
        const result = yield crawlWebPage(url);
        if (result) {
            const dom = new JSDOM(result.html);
            const document = dom.window.document;
            const rootElement = document.body;
            const figmaTree = parseHTMLToFigmaNode(rootElement);
            console.log("Generated Figma-like JSON:");
            console.log(JSON.stringify(figmaTree, null, 2));
        }
    });
}
main();
