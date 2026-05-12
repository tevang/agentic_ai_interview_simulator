from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.client = OpenAI()

    def text(self, *, model: str, system: str, user: str, temperature: float = 0.2) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._supports_temperature(model):
            payload["temperature"] = temperature

        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content

    def json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        schema_name: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            }
        }
        if self._supports_temperature(model):
            payload["temperature"] = temperature

        response = self.client.chat.completions.create(**payload)
        return json.loads(response.choices[0].message.content)

    @staticmethod
    def _supports_temperature(model: str) -> bool:
        return not model.startswith("gpt-5")
