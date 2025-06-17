 

import httpx
import asyncio

# ---------------------------
# Agent Class
# ---------------------------
class Agent:
    def __init__(self, name: str, instructions: str, model):
        self.name = name
        self.instructions = instructions
        self.model = model

    def describe(self):
        return f"Agent '{self.name}' initialized with instructions: {self.instructions}"


# ---------------------------
# AsyncOpenAI Class (Gemini API Wrapper)
# ---------------------------
class AsyncOpenAI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    async def query(self, prompt: str):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "gemini-1.5-flash",  # ya ap apna model_name pass karen
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


# ---------------------------
# OpenAIChatCompletionsModel (Model Wrapper)
# ---------------------------
class OpenAIChatCompletionsModel:
    def __init__(self, model_name: str = "gemini-1.5-flash", openai_client=None):
        self.model_name = model_name
        self.client = openai_client

    async def complete(self, prompt: str):
        return await self.client.query(prompt)


# ---------------------------
# RunConfig (Optional Configs)
# ---------------------------
class RunConfig:
    def __init__(self, model=None, model_provider=None, tracing_disabled: bool = True):
        self.model = model
        self.model_provider = model_provider
        self.tracing_disabled = tracing_disabled
        self.max_tokens = 1000
        self.temperature = 0.7


# ---------------------------
# Runner Class (For running Agent)
# ---------------------------
class Runner:
    @staticmethod
    def run_sync(agent: Agent, prompt: str, run_config: RunConfig):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        output = loop.run_until_complete(agent.model.complete(prompt))
        return output
