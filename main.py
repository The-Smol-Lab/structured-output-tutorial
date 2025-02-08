#main.py from https://github.com/krittaprot/structured-output-tutorial

import time
from datetime import datetime
from enum import Enum
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from config import GEMINI_API_KEY
from loader import Loader

import argparse

MODEL_CONFIGS = {
    'lmstudio': {
        'openai_api_base': "http://localhost:1234/v1",
        'openai_api_key': "lm-studio",
        'model_name': "deepseek-r1-distill-qwen-32b"
    },
    'ollama': {
        'openai_api_base': "http://localhost:11434/v1",
        'openai_api_key': "ollama",
        'model_name': "deepseek-r1:14b"
    },
    'gemini': {
        'openai_api_base': "https://generativelanguage.googleapis.com/v1beta/openai/",
        'openai_api_key': GEMINI_API_KEY,
        'model_name': "gemini-2.0-flash-exp"
    }
}

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"


class StockSentiment(BaseModel):
    company_name: str = Field(
        ..., 
        description="Company name with ticker symbol, e.g., NVIDIA Corporation (NVDA)"
    )
    justification: str = Field(
        ..., 
        description="Detailed explanation with specific numbers from the article"
    )
    sentiment: SentimentLabel = Field(
        ..., 
        description="Sentiment classification based on content analysis"
    )
    confidence: float = Field(
        ..., 
        description="Confidence level between 0 and 1", 
        ge=0, 
        le=1
    )

    @field_validator("company_name")
    def validate_company_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Company name cannot be empty")
        if len(v) > 100:
            raise ValueError("Company name must be ≤ 100 characters")
        return v

    @field_validator("confidence", mode="before")
    def normalize_confidence(cls, v: float) -> float:
        # Convert percentage values to decimal
        if isinstance(v, (int, float)) and v > 1:
            v /= 100
        return round(v, 2)

    @field_validator("justification")
    def validate_justification(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Justification cannot be empty")
        if any(char in v for char in ("±", "≈")):
            raise ValueError("Use exact numbers from article")
        return v


class NewsSentiment(BaseModel):
    stocks: List[StockSentiment] = Field(
        ...,
        examples=[[{
            "company_name": "NVIDIA Corporation (NVDA)",
            "sentiment": "positive",
            "confidence": 0.95,
            "justification": "Q4 revenue increased 15% to $22.1B driven by AI chips"
        }]]
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Analysis timestamp in ISO format"
    )

    @field_validator("stocks")
    def validate_stocks(cls, v: List[StockSentiment]) -> List[StockSentiment]:
        if not v:
            raise ValueError("At least one stock required")
        return v


def main() -> None:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run news sentiment analysis with a specified mode and model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gemini", "lmstudio", "ollama"],
        default="gemini",
        help="Mode to run the script in: gemini, lmstudio, or ollama (default: gemini)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override the default model name for the selected mode"
    )
    args = parser.parse_args()

    # Validate and load configuration
    mode = args.mode
    config = MODEL_CONFIGS.get(mode)
    if not config:
        raise ValueError(f"Invalid mode: {mode}. Choose from: {list(MODEL_CONFIGS.keys())}")

    # Override the model_name if provided via command line
    if args.model_name:
        config["model_name"] = args.model_name

    # Print selected configuration
    print(f"Running in mode: {mode}")
    print(f"Using model: {config['model_name']}")

    # Initialize the Chat model
    model = ChatOpenAI(
        model_name=config['model_name'],
        openai_api_base=config['openai_api_base'],
        openai_api_key=config['openai_api_key'],
        temperature=0  # Deterministic output
    )
    structured_llm = model.with_structured_output(NewsSentiment)

    # Define the system prompt
    system_prompt = """You are a senior financial analyst specializing in news sentiment analysis:
    1. Identify all publicly traded companies in the text
    2. For each company, determine sentiment using:
    - Financial performance metrics (exact figures/percentages)
    - Strategic developments (mergers, partnerships)
    - Regulatory/legal changes
    - Market reactions (stock moves, analyst ratings)

    3. Provide confidence scores as decimal values between 0 and 1 (e.g., 0.85 for 85% confidence).
    Never use percentage values for confidence scores.

    Include specific numerical data, quantified impacts, and precise metrics.
    Confidence scores must reflect evidence strength. Never invent information."""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze this {current_date} news article:\n\n{article}")
    ])

    # Load the article content
    try:
        with open('content.txt', 'r', encoding='utf-8') as file:
            article = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("The file 'content.txt' was not found. Please ensure it exists.")

    # Prepare the input data
    current_date = datetime.now().strftime("%Y-%m-%d")
    input_data = {"article": article, "current_date": current_date}

    # Process the article and measure execution time
    start_time = time.time()
    with Loader("Processing article..."):
        result = (prompt | structured_llm).invoke(input_data)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\nModel Used: {config['model_name']}")
    print(f"Analysis timestamp: {result.timestamp}")
    print(f"Time taken to process the article: {elapsed_time:.2f} seconds")

    for stock in result.stocks:
        print("\n" + "*" * 50)
        print(f"Company: {stock.company_name}")
        print(f"Sentiment: {stock.sentiment.value}")
        print(f"Confidence: {stock.confidence:.0%}")
        print(f"Justification: {stock.justification}")


if __name__ == "__main__":
    main()