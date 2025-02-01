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
git clone https://github.com/krittaprot/structured-output-tutorial.git
cd structured-output-tutorial
python -m venv .llm_env
cd .llm_env
Scripts/activate (for windows) or source bin/activate (for mac)
cd ..
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
system_prompt = """
               You are a senior financial analyst specializing in news sentiment analysis:
               1. Identify all publicly traded companies in the text
               2. For each company, determine sentiment using:
               - Financial performance metrics (exact figures/percentages)
               - Strategic developments (mergers, partnerships)
               - Regulatory/legal changes
               - Market reactions (stock moves, analyst ratings)

               3. Provide confidence scores as decimal values between 0 and 1 (e.g., 0.85 for 85% confidence).
               Never use percentage values for confidence scores.

               Include specific numerical data, quantified impacts, and precise metrics.
               Confidence scores must reflect evidence strength. Never invent information.
               """

user_prompt =   """ 
                    The current date is {current_date}, 
                    analyze the following article and provide sentiment analysis for each publicly traded company 
                    mentioned in the text below:
                    {article}
                """
```

## Usage

Run gemini model as default
```python
python main.py
or
python main.py --mode gemini #default to "gemini-2.0-flash-exp"
or
python main.py --mode gemini --model_name "gemini-2.0-flash-exp"
```

Pass in mode and model_name to run a specific model of choice from lmstudio or ollama
```python
python main.py --mode lmstudio --model_name "deepseek-r1-distill-qwen-32b@iq2_s"
python main.py --mode ollama --model_name "deepseek-r1:14b"
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
- All Gemini Models (e.g., `gemini-2.0-flash-exp`, `gemini-2.0-flash-thinking-exp-01-21`, `gemini-exp-1206`, and etc.)
- Any OpenAI-compatible API endpoint including Ollama/LM Studio

## License

MIT License - See [LICENSE](LICENSE) for details

## Author

Maintained by [Krittaprot Tangkittikun](https://www.linkedin.com/in/krittaprot-tangkittikun-0103a9109/).
