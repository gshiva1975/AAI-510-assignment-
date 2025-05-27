
from a2a.agent import Agent
from a2a.schema import ToolDefinition
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class TwitterAgent(Agent):
    def __init__(self):
        super().__init__("twitter_sentiment")
        self.sentiment_model = pipeline("sentiment-analysis")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        df = pd.read_csv("twitter_validation.csv")
        df.columns = ['id', 'source', 'label', 'text']
        corpus = df["text"].dropna().astype(str).tolist()[:50]
        self.corpus = corpus
        self.embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

    async def onInit(self):
        return [ToolDefinition(name="analyze_prompt", parameters={"text": {"type": "string"}})]

    async def analyze_prompt(self, text: str):
        sentiment = self.sentiment_model(text[:512])[0]
        query_emb = self.embedder.encode(text, convert_to_tensor=True)
        match = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        top = match.argmax().item()
        return {"text": text, "sentiment": sentiment["label"], "similar_to": self.corpus[top]}

if __name__ == "__main__":
    import asyncio
    asyncio.run(TwitterAgent().run())
