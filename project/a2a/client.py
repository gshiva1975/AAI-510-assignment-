class A2AClient:
    async def send(self, prompt):
        class Response:
            def __init__(self, content):
                self.content = f"[✓ iphone_sentiment] POSITIVE: {prompt}"
        return Response(content=prompt)
