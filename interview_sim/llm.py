from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.client = OpenAI()

    def text(self, *, model: str, system: str, user: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
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
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content)