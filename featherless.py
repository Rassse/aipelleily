from openai import OpenAI

client = OpenAI(
  base_url="https://api.featherless.ai/v1",
  api_key="rc_e638e8f58709e1d707c7704a3343b78f8121b2e0025710fa15dd8641d1daa0a3",
)

response = client.chat.completions.create(
  model='Qwen/Qwen2.5-7B-Instruct',
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
)
print(response.choices[0].message.content)