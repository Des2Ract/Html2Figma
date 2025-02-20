import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { parse } from './parser.js';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(cors());

interface GenerateJsonRequestBody {
  url: string;
}

class ApiController {
  static async generateJsonHandler(req: Request, res: Response): Promise<void> {
    const { url } = req.body;

    if (!url) {
      res.status(400).json({ error: 'URL is required' });
      return;
    }

    try {
      console.log(`Processing URL: ${url}`);
      const figmaTree = await parse(url);
      res.json({ json: figmaTree });
    } catch (error) {
      console.error('Error processing URL:', error);
      res.status(500).json({ error: 'Failed to generate JSON' });
    }
  }
}

app.post('/generate-json', ApiController.generateJsonHandler);

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
