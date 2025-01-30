# News Sentiment Analyzer

A robust system for analyzing market sentiment in news articles using AI and structured data validation.

![image](https://github.com/user-attachments/assets/590ffeaa-ecc6-4322-a44e-25f1b211e95e)

## Features

- **Company Identification**: Automatically detects publicly traded companies in news content
- **Sentiment Classification**: Categorizes sentiment as Positive, Negative, or Mixed
- **Confidence Scoring**: Provides numerical confidence levels (0-1) for each analysis
- **Data Validation**: Ensures:
  - Valid company names with ticker symbols
  - Confidence scores within proper ranges
  - Exact numerical references in justifications
  - ISO-8601 timestamps
- **Structured Output**: Returns validated JSON data with Pydantic models

## Installation

```bash
git clone git+https://github.com/krittaprot/structured-output-tutorial.git
cd structured-output-tutorial
pip install -r requirements.txt
```

## Configuration

1. Get a [Google Gemini API key](https://aistudio.google.com/app/apikey)
2. Create `config.py` with:
```python
GEMINI_API_KEY = "your_api_key_here"
```

## Usage

```python
python run main.py
```

## Example Output

```
Model Used: gemini-2.0-flash-exp
Analysis timestamp: 2024-01-09T15:34:56.789012
Time taken to process the article: 2.45 seconds

**************************************************
Company: Meta
Sentiment: positive
Confidence: 85%
Justification: Recent reports highlight Meta's significant advancements and investments in artificial intelligence (AI). The company plans to invest hundreds of billions over the long term in AI infrastructure, aiming to make Meta AI accessible to over a billion users by 2025.....
```

## Model Support

Currently compatible with:
- Gemini 1.5 Flash/Pro (`gemini-2.0-flash-exp`)
- Any OpenAI-compatible API endpoint
- Local models via LM Studio

## Validation Rules

| Field          | Validation Criteria                     |
|----------------|-----------------------------------------|
| Company Name   | Non-empty, ≤100 characters              |
| Confidence     | 0-1 range, rounded to 2 decimals        |
| Justification  | Requires exact numbers, no approximations |
| Timestamp      | Auto-generated ISO-8601 format          |

## License

MIT License - See [LICENSE](LICENSE) for details
