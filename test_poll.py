import requests

payload = {
    "messages": [{"role": "user", "content": "Hola"}],
    "model": "mistral-large"
}
response = requests.post("https://text.pollinations.ai/openai", json=payload)
print(response.status_code)
print(response.text)
