from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
from datetime import datetime
import time

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class StockSentiment(BaseModel):
    company_name: str = Field(..., description="The name of the company", example="NVIDIA Corporation (NVDA)")
    sentiment: SentimentLabel = Field(..., description="either positive, neutral, or negative")
    confidence: float = Field(..., description="confidence level of the sentiment analysis between 0 and 1", example=0.95)
    justification: str = Field(..., description="justification with specific numbers from article")  # Changed from Optional to required
    
    @field_validator("company_name")
    def validate_company_name(cls, v):
        if not v.strip():
            raise ValueError("Company name cannot be empty")
        if len(v) > 100:
            raise ValueError("Company name must be 100 characters or fewer")
        return v

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 2)  # Round to 2 decimal places for cleaner output

    @field_validator("justification")
    def validate_justification(cls, v):
        if not v.strip():
            raise ValueError("Justification cannot be empty")
        if "±" in v or "≈" in v:  # Prevent approximations
            raise ValueError("Use exact numbers from article, not approximations")
        return v

class NewsSentiment(BaseModel):
    stocks: List[StockSentiment] = Field(
            ...,
            example=[
                {
                    "company_name": "NVIDIA Corporation (NVDA)", 
                    "sentiment": "positive", 
                    "confidence": 0.95, 
                    "justification": "Q4 revenue increased 15% to $22.1 billion driven by AI chip demand"
                },
                {
                    "company_name": "Tesla, Inc. (TSLA)", 
                    "sentiment": "negative", 
                    "confidence": 0.85, 
                    "justification": "Vehicle deliveries dropped 8.5% to 435,000 units in Q3"
                }
            ]
        )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the analysis in ISO format"
    )

    @field_validator("stocks")
    def validate_stocks(cls, v):
        if not v:
            raise ValueError("Stocks list cannot be empty")
        return v

# Initialize Chat model
model = ChatOpenAI(
    model_name="fuseo1-deepseekr1-qwen2.5-coder-32b-preview-v0.1",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.3
)

# Add structured output capability
structured_llm = model.with_structured_output(NewsSentiment)

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


user_prompt = """Analyze this news article dated {current_date}:

{article}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", user_prompt)
])

# Create processing chain
chain = prompt | structured_llm

# Example usage
article = """

TMBThanachart Bank Public Company Limited (SET: TTB) has announced a share repurchase program for 2025-2027 with a total
budget of 21 billion baht. In the rst round in 2025, TTB will repurchase no more than 3.5 billion shares (equivalent to 3.6% of total
outstanding shares) with a total budget of no more than 7 billion
baht.
This announcement implies an average price of THB2.00 per share,
which is 4.7% higher than the closing price of TTB on Tuesday at
THB1.91 per share. The program will start from February 3 to August
1.
The share repurchase is common in the market, however, the magnitude of the purchasing power from TTB is far exceeding other commercial banks that had done the same program in the past. In 2020,
Kasikornbank Public Company Limited (SET: KBANK) announced a
share repurchase program that accounted to 1% of its total outstanding shares, while Kiatnakin Phatra Bank Public Company Limited
(SET: KKP) had also done the same last year for a total of 2.6% of its
total outstanding shares.
According to Kiatnakin Phatra Securities, this is a positive momentum
for TTB’s share price, and it can enable TTB’s 2025 RoE to improve
by roughly 13 bps to 8.6% in 2025, while the CET-1 ratio will remain
around 17%.
Additionally, the Board of Directors of TTB also resolved to approve
the 2025 Employee Joint Investment Program (EJIP2025) to serve as
1/30/25, 8:53 AM TTB Announces Aggressive THB21 Billion Share Buyback with EJIP Program up to 750% Returns - KAOHOON INTERNATIONAL
https://www.kaohooninternational.com/markets/551564 2/4
a long-term incentive for employees which builds a sense of ownership in the organization, an employee retention tool, as well as an alternative to traditional compensation options.
The program is also not intended to allow directors or executives to
interfere with the investment. Moreover, TTB stated that the bank
will contribute 100%-750% of participating employee’s contributions
with a total budget of 841 million baht.


"""

# Update example usage with current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Start the timer
start_time = time.time()

print('Processing article...')
result = chain.invoke({"article": article, "current_date": current_date})

# Calculate the elapsed time
elapsed_time = time.time() - start_time

print(f"Analysis timestamp: {result.timestamp}")
print(f"Time taken to process the article: {elapsed_time:.2f} seconds")

for stock in result.stocks:
    print(f"Company: {stock.company_name}")
    print(f"Sentiment: {stock.sentiment.value}")
    print(f"Confidence: {stock.confidence:.0%}")  # Format as percentage
    print(f"Justification: {stock.justification}\n")


