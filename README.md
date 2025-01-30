# Structuring the Unstructured

An AI system example use case for analyzing market sentiment in news articles with explanation using structured output.

![image](https://github.com/user-attachments/assets/590ffeaa-ecc6-4322-a44e-25f1b211e95e)

## Features

- **Company Identification**: Automatically detects publicly traded companies in news content
- **Sentiment Classification**: Categorizes sentiment as Positive, Negative, or Mixed
- **Confidence Scoring**: Provides numerical confidence levels (0-1) for each analysis
- **Sentiment Justification**: Provides explanation of how the AI came up with the sentiment label based on your provided content.
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

## System & User Prompt

```python
# Create prompt template with detailed system message
system_prompt = """You are a senior financial analyst with expertise in news sentiment analysis. 
When analyzing articles, follow these guidelines:

1. Identify all publicly traded companies mentioned in the text
2. For each company, determine market sentiment based on:
   - Explicit statements about financial performance (include exact figures/percentages)
   - Strategic developments (mergers, partnerships, innovations)
   - Regulatory/legal implications
   - Market reactions (stock movements, analyst ratings)

For each sentiment determination:
- Include SPECIFIC NUMERICAL DATA from the article when available (revenue figures, percentage changes, booking numbers)
- State QUANTIFIED IMPACTS ("9% revenue growth" not just "revenue growth")
- Mention EXACT TIME REFERENCES ("Q4 2023" not just "recently")
- Use PRECISE METRICS from the text ($27.35 billion, 6% stock increase)

Maintain strict requirements:
- Confidence scores must reflect article evidence strength
- Never invent information not explicitly stated
- Use exact company names with ticker symbols
- Prioritize recent information when multiple data points exist"""

user_prompt =   """ 
                    The current date is {current_date}, 
                    analyze the following article and provide sentiment analysis for each publicly traded company 
                    mentioned in the text below:
                    {article}
                """
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
- All Gemini Models (e.g., `gemini-2.0-flash-exp`)
- Any OpenAI-compatible API endpoint including Ollama/LM Studio

## License

MIT License - See [LICENSE](LICENSE) for details
