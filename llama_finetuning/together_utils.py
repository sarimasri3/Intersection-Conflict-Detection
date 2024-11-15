# llama_finetuning/together_utils.py

"""
Together AI Utility Functions

This module contains helper functions for interacting with the Together AI API and the LLAMA model.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import requests
import json
from dotenv import load_dotenv, find_dotenv


def load_env():
    """
    Loads environment variables from a .env file.
    """
    load_dotenv(find_dotenv())


def llama32(messages, model_size=11):
    """
    Sends a chat completion request to the LLAMA model via the Together AI API.

    Args:
        messages (list): List of messages in the conversation.
        model_size (int): Size of the LLAMA model (e.g., 11 for 11B model).

    Returns:
        str: The content of the assistant's reply.
    """
    load_env()
    model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
    url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.0,
        "stop": ["<|eot_id|>", "<|eom_id|>"],
        "messages": messages
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    res = response.json()

    if 'error' in res:
        raise Exception(res['error'])

    return res['choices'][0]['message']['content']
