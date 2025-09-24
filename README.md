# Presentation Narration Generator

Generate structured presentation narration and slide content using LearnLM.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## Usage

```bash
python simple_example.py
```

## API

```python
from src.models import Summary
from src.generator import NarrationGenerator

summary = Summary(
    title="Your Topic",
    content="Your content...",
    estimated_duration=3
)

generator = NarrationGenerator()
result = generator.generate(summary, generate_slides=True)
generator.export_results(result)
```

## Output

- `output/narration_script.txt` - Complete script
- `output/presentation_data.json` - Structured data

Each slide contains:
- Narration text
- JSON structure with heading and bullet points 