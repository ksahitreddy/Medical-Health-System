import requests
import json
import os
from transformers import AutoTokenizer

# 🆕 Replace this with your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"   # or "llama3-70b-8192"
SYSTEM_PROMPT = "You are a medical chatbot specialized in all fields of medicine. Answer only medical or clinical related queries with medically accurate information. If a question is unrelated, politely inform the user that you can only answer medical-related questions. Provide a clear, concise, and accurate medical response"
MAX_CONTEXT_TOKENS = 4096

# Use HF tokenizer to count tokens
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def count_tokens(messages):
    combined = "".join([f"{m['role']}: {m['content']}\n" for m in messages])
    return len(tokenizer.encode(combined))

def query_local_llm(prompt, history=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in reversed(history):
            if count_tokens(messages + [msg, {"role": "user", "content": prompt}]) < MAX_CONTEXT_TOKENS:
                messages.insert(1, msg)
            else:
                break

    # Truncate the prompt if it exceeds the max token limit
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) > MAX_CONTEXT_TOKENS:
        prompt_tokens = prompt_tokens[:MAX_CONTEXT_TOKENS]
        prompt = tokenizer.decode(prompt_tokens)

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 6000,
        "temperature": 0.4,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return f"🚨 Error parsing API response: {str(e)}"
    except requests.exceptions.HTTPError as e:
        error_msg = f"🚨 HTTP Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg += f"\n🔍 Details: {error_data.get('error', {}).get('message', str(error_data))}"
            except:
                error_msg += f"\n🔍 Response: {e.response.text[:500]}..."  # Truncate long responses
        return error_msg
    except requests.exceptions.RequestException as e:
        return f"🚨 Request failed: {str(e)}"
    except Exception as e:
        return f"🚨 Unexpected error: {str(e)}"

