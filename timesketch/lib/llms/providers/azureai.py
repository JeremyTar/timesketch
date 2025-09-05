"""Azure Ai Foundry LLM provider."""

from timesketch.lib.llms.providers import interface, manager
import requests

has_required_deps = True
try:
    from openai import AzureOpenAI
except ImportError:
    has_required_deps = False

class AzureAI(interface.LLMProvider):
    NAME = "azureai"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.endpoint = self.config.get("endpoint")
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.api_version = self.config.get("api_version", "2024-02-15-preview")
        if not self.endpoint or not self.api_key or not self.model:
            raise ValueError("endpoint, api_key, and model are required for AzureAI provider")

    def generate(self, prompt, response_schema=None):
        url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.get("max_output_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "top_p": self.config.get("top_p", 0.95)
        }
        timeout = self.config.get("timeout", 60)

        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

if has_required_deps:
    manager.LLMManager.register_provider(AzureAI)
