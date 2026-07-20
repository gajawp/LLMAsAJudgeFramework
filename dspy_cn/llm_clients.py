import os
import os
import json
from openai import OpenAI
from anthropic import Anthropic


class BaseJudgeClient:
    def generate_json(self, prompt: str) -> dict:
        raise NotImplementedError


class OpenAIJudgeClient(BaseJudgeClient):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_json(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=700,
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)


class ClaudeJudgeClient(BaseJudgeClient):
    def __init__(self, model: str):
        self.model = model
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate_json(self, prompt: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        content = response.content[0].text.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)


def get_judge_client(provider: str, model: str):
    provider = provider.lower()

    if provider == "openai":
        return OpenAIJudgeClient(model)

    if provider in ["anthropic", "claude"]:
        return ClaudeJudgeClient(model)

    raise ValueError(f"Unsupported judge provider: {provider}")