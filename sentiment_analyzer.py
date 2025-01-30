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
    justification: Optional[str] = Field(None, description="justification for the sentiment label")

    @field_validator("company_name")
    def validate_company_name(cls, v):
        if not v.strip():
            raise ValueError("Company name cannot be empty")
        if len(v) > 100:
            raise ValueError("Company name must be 100 characters or fewer")
        return v

class NewsSentiment(BaseModel):
    stocks: List[StockSentiment] = Field(
        ...,
        description="List of stocks sentiment with justification",
        example=[
            {"company_name": "NVIDIA Corporation (NVDA)", "sentiment": "positive", "justification": "Strong earnings report"},
            {"company_name": "Tesla, Inc. (TSLA)", "sentiment": "negative", "justification": "Recall of vehicles"}
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
    model_name="deepseek-r1-distill-llama-8b",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.1
)

# Add structured output capability
structured_llm = model.with_structured_output(NewsSentiment)

# Create prompt template (no format instructions needed)
prompt = ChatPromptTemplate.from_template(
    """Analyze the following news article and extract stock sentiments:

{article}"""
)

# Create processing chain
chain = prompt | structured_llm

# Example usage
article = """

Stock Market Update: Tesla Surges, Microsoft Holds Steady, and Rivian Faces Headwinds

By Tom RussyJanuary 29, 2025

The stock market saw mixed movements today as investors reacted to earnings reports and broader economic indicators. While some tech giants continued their upward trajectory, others faced mounting challenges.

Tesla (TSLA) Soars on Strong Q4 Results

Tesla Inc. (NASDAQ: TSLA) saw its stock price surge by over 8% after reporting stronger-than-expected fourth-quarter earnings. The electric vehicle (EV) giant posted a revenue of $29.3 billion, surpassing Wall Street estimates, and reported a net profit increase of 12% year-over-year. CEO Elon Musk highlighted the company's success in ramping up production at its new Gigafactories and expanding its full self-driving (FSD) capabilities.

"We are seeing accelerating demand for our vehicles, particularly in China and Europe," Musk stated during the earnings call. Analysts have responded positively, with several firms upgrading their price targets for the stock, citing Tesla's strong delivery numbers and advancements in AI-driven autonomous driving technology.

Microsoft (MSFT) Holds Steady Amid Cloud Strength

Microsoft Corporation (NASDAQ: MSFT) remained largely unchanged, closing with a modest 0.5% gain. The company’s cloud segment, led by Azure, continued to perform well, reporting a 21% increase in revenue. However, growth in its traditional software business showed signs of slowing.

"Microsoft’s strong cloud growth is offsetting slower gains in other areas, and the stock remains a safe bet for long-term investors," said David Klein, an analyst at JP Morgan.

While AI investments and enterprise demand remain key drivers, investors appear to be taking a wait-and-see approach ahead of next quarter’s guidance. Some analysts believe that competition from Amazon Web Services (AWS) and Google Cloud could temper Azure’s future expansion.

Rivian (RIVN) Struggles Amid Production Challenges

On the downside, Rivian Automotive Inc. (NASDAQ: RIVN) faced a 6% decline after missing delivery targets and cutting its production forecast for the year. Supply chain disruptions and higher-than-expected costs have continued to weigh on the EV startup, despite strong demand for its electric trucks and SUVs.

CEO RJ Scaringe acknowledged the difficulties in a shareholder letter, stating, "We are actively addressing supply constraints and optimizing our manufacturing processes, but challenges persist."

Analysts are growing concerned about Rivian’s cash burn rate and the need for additional capital raises. Some remain optimistic about the company’s long-term potential but warn that near-term volatility is likely.

Conclusion

With Tesla riding high on strong earnings, Microsoft maintaining stability through cloud growth, and Rivian struggling with production issues, the stock market continues to reflect a mixed bag of sentiment. Investors will be watching closely for further developments as companies navigate the evolving economic landscape.




"""

# Start the timer
start_time = time.time()

print('Processing article...')
result = chain.invoke({"article": article})

# Calculate the elapsed time
elapsed_time = time.time() - start_time

print(f"Analysis timestamp: {result.timestamp}")
print(f"Time taken to process the article: {elapsed_time:.2f} seconds")

for stock in result.stocks:
    print(f"Company: {stock.company_name}")
    print(f"Sentiment: {stock.sentiment.value}")
    print(f"Justification: {stock.justification}\n")