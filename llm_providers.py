# llm_providers.py
import os
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
import anthropic
from mistralai import Mistral

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text response from the LLM"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the provider"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is properly configured"""
        pass

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST")
        self.model = os.getenv("OLLAMA_MODEL")

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            return f"[Error contacting Ollama: {e}]"

    @property
    def display_name(self) -> str:
        return "Ollama (Local LLMs)"

    @property
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class OpenAIProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def generate_text(self, prompt: str) -> str:
        if not self.client:
            return "[Error: OpenAI API key not configured]"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the more affordable model
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1800,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error contacting OpenAI: {e}]"

    @property
    def display_name(self) -> str:
        return "OpenAI (ChatGPT)"

    @property
    def is_available(self) -> bool:
        return self.client is not None

class ClaudeProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None

    def generate_text(self, prompt: str) -> str:
        if not self.client:
            return "[Error: Anthropic API key not configured]"

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Using the faster, cheaper model
                max_tokens=1800,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"[Error contacting Claude: {e}]"

    @property
    def display_name(self) -> str:
        return "Claude (Anthropic)"

    @property
    def is_available(self) -> bool:
        return self.client is not None

class MistralProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            self.client = Mistral(api_key=api_key)
        else:
            self.client = None

    def generate_text(self, prompt: str) -> str:
        if not self.client:
            return "[Error: Mistral API key not configured]"

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                max_tokens=1800,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error contacting Mistral: {e}]"

    @property
    def display_name(self) -> str:
        return "Mistral (Le Chat)"

    @property
    def is_available(self) -> bool:
        return self.client is not None

# Registry of all available providers
LLM_PROVIDERS = {
    "ollama": OllamaProvider(),
    "openai": OpenAIProvider(),
    "claude": ClaudeProvider(),
    "mistral": MistralProvider(),
}

def get_available_providers():
    """Return a dict of providers that are properly configured"""
    available_providers = {key: provider for key, provider in LLM_PROVIDERS.items() if provider.is_available}
    if os.getenv("OLLAMA_HOST"):
        if os.getenv("OLLAMA_MODEL"):
            available_providers["ollama"] = OllamaProvider()
    return available_providers


def get_provider(provider_name: str) -> LLMProvider:
    """Get a specific provider by name"""
    if provider_name not in LLM_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")

    provider = LLM_PROVIDERS[provider_name]
    if not provider.is_available:
        raise ValueError(f"Provider '{provider_name}' is offline or not properly configured")

    return provider
