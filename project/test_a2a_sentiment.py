
import asyncio
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from a2a.client import A2AClient

TEST_DATA = [
    {"prompt": "The iPhone battery drains too fast.", "expected_agent": "iphone_sentiment", "expected_sentiment": "NEGATIVE"},
    {"prompt": "The new iPhone design is stunning!", "expected_agent": "iphone_sentiment", "expected_sentiment": "POSITIVE"},
    {"prompt": "Twitter keeps crashing on my phone.", "expected_agent": "twitter_sentiment", "expected_sentiment": "NEGATIVE"},
    {"prompt": "I had an amazing time using Twitter Spaces.", "expected_agent": "twitter_sentiment", "expected_sentiment": "POSITIVE"},
    {"prompt": "Why are people unhappy with the new iPhone?", "expected_agent": "iphone_sentiment", "expected_sentiment": "NEGATIVE"},
    {"prompt": "Twitter is full of trolls lately.", "expected_agent": "twitter_sentiment", "expected_sentiment": "NEGATIVE"},
    {"prompt": "Are iPhone users complaining on Twitter?", "expected_agent": "twitter_sentiment", "expected_sentiment": "NEGATIVE"},
    {"prompt": "I saw a tweet saying iPhones overheat easily.", "expected_agent": "twitter_sentiment", "expected_sentiment": "NEGATIVE"}
]

async def test_and_evaluate():
    client = A2AClient()
    correct_routing, correct_sentiment = 0, 0
    results = []

    for entry in TEST_DATA:
        prompt = entry["prompt"]
        print(f"\n🧪 Testing Prompt: {prompt}")
        response = await client.send(prompt)
        print(f"➡️  Response: {response.content}")

        predicted = response.content
        routed_agent = predicted.split("[✓ ")[1].split("]")[0].strip()
        predicted_sentiment = predicted.split("] ")[1].split(":")[0].strip()

        route_match = routed_agent == entry["expected_agent"]
        sentiment_match = predicted_sentiment.upper() == entry["expected_sentiment"].upper()

        if route_match:
            correct_routing += 1
        if sentiment_match:
            correct_sentiment += 1

        results.append({
            "prompt": prompt,
            "expected_agent": entry["expected_agent"],
            "predicted_agent": routed_agent,
            "expected_sentiment": entry["expected_sentiment"],
            "predicted_sentiment": predicted_sentiment,
            "route_match": route_match,
            "sentiment_match": sentiment_match
        })

    total = len(TEST_DATA)
    print("\n✅ Evaluation Metrics:")
    print(f"Routing Accuracy: {correct_routing}/{total} = {correct_routing/total:.2f}")
    print(f"Sentiment Accuracy: {correct_sentiment}/{total} = {correct_sentiment/total:.2f}")

    df = pd.DataFrame(results)
    df.to_csv("a2a_sentiment_eval_results.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(x="prompt", y="route_match", data=df, palette="coolwarm")
    plt.title("Routing Accuracy per Prompt")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("routing_accuracy_chart.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.barplot(x="prompt", y="sentiment_match", data=df, palette="crest")
    plt.title("Sentiment Accuracy per Prompt")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("sentiment_accuracy_chart.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(test_and_evaluate())
