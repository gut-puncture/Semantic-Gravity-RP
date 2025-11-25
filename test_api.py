from openai import OpenAI
import os

key = "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"
client = OpenAI(api_key=key)

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print("API Success:", response.choices[0].message.content)
except Exception as e:
    print("API Failed:", e)
