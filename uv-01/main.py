
from dotenv import load_dotenv
import os
from Agent import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner  

# Load API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Create external client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Setup model
model = OpenAIChatCompletionsModel(
    model_name="gemini-1.5-flash",
    openai_client=external_client
)

# Setup config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create agent
agent = Agent(
    name="Translator",
    instructions="Translate the given text to the specified language.",
    model=model
)

# Run agent
result = Runner.run_sync(agent, "Translate 'Hello everyone' to Urdu", config)

# Output
print("\nCALLING AGENT\n")
print(result)
