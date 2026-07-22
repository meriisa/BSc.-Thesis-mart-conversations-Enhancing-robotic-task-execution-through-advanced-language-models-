# llm_provider.py

import requests
import openai
import rospy


class BaseLLMProvider:
    def chat(self, system_message: str, user_prompt: str) -> str:
        raise NotImplementedError


# -------------------------
# Provider Implementations
# -------------------------

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        openai.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            rospy.logerr(f"OpenAI API error: {e}")
            return "I'm sorry, I couldn't process that request."


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            endpoint = "https://api.deepseek.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            rospy.logerr(f"DeepSeek API error: {e}")
            return "I'm sorry, I couldn't process that request."


class LlamaCppProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key_unused,
        model_name_unused,
        max_tokens,
        temperature
    ):
        self.endpoint = rospy.get_param(
            "models/llm_endpoint",
            "http://localhost:8000/completion"
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

        self.timeout = rospy.get_param(
            "models/llm_timeout_seconds",
            30
        )

    def chat(self, system_message, user_prompt):
        try:
            payload = {
                "prompt": f"{system_message}\n{user_prompt}",
                "n_predict": self.max_tokens,
                "temperature": self.temperature
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            return data.get("content", "").strip()

        except Exception as e:
            rospy.logerr(f"Llama.cpp API error: {e}")
            return "I'm sorry, I couldn't process that request."


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key_unused,
        model_name,
        max_tokens,
        temperature
    ):
        base_endpoint = rospy.get_param(
            "models/llm_endpoint",
            "http://172.17.0.1:11434"
        )

        self.endpoint = base_endpoint.rstrip("/") + "/api/chat"
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.timeout = rospy.get_param(
            "models/llm_timeout_seconds",
            120
        )

    def chat(self, system_message, user_prompt):
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            if "message" not in data:
                raise ValueError(
                    f"Unexpected Ollama response: {data}"
                )

            return data["message"]["content"].strip()

        except Exception as e:
            rospy.logerr(f"Ollama API error: {e}")
            return "I'm sorry, I couldn't process that request."


class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            endpoint = "https://api.anthropic.com/v1/messages"

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_message,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }

            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            return data["content"][0]["text"].strip()

        except Exception as e:
            rospy.logerr(f"Claude API error: {e}")
            return "I'm sorry, I couldn't process that request."


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key,
        model_name_unused,
        max_tokens_unused,
        temperature_unused
    ):
        self.api_key = api_key

    def chat(self, system_message, user_prompt):
        try:
            endpoint = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/gemini-pro:generateContent?key={self.api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": system_message
                            },
                            {
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            return (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )

        except Exception as e:
            rospy.logerr(f"Gemini API error: {e}")
            return "I'm sorry, I couldn't process that request."


# -------------------------
# Factory Loader
# -------------------------

class LLMProviderFactory:
    provider_classes = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "llama.cpp": LlamaCppProvider,
        "ollama": OllamaProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider
    }

    @classmethod
    def create_provider(
        cls,
        provider_name,
        api_key,
        model_name,
        max_tokens,
        temperature
    ):
        if provider_name not in cls.provider_classes:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}"
            )

        provider_class = cls.provider_classes[provider_name]

        return provider_class(
            api_key,
            model_name,
            max_tokens,
            temperature
        )