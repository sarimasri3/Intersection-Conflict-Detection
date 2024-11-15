# run_llama_fine_tuning.py

"""
Script to Fine-Tune LLAMA Model using Together AI API

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import pandas as pd
from llama_finetuning.prepare_data import create_finetune_dataset, verify_dataset
from llama_finetuning.fine_tune_llama import fine_tune_model, monitor_fine_tuning_job
from together import Together

# Set your Together AI API key
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("Please set your Together AI API key as an environment variable 'TOGETHER_API_KEY'.")

# Load datasets
train_df = pd.read_csv('data/train_set.csv')
val_df = pd.read_csv('data/val_set.csv')

# Define system prompt
system_prompt = """
You are an Urban Intersection Traffic Conflict Detector, responsible for monitoring a four-way intersection with traffic coming from the north, east, south, and west. Each direction has two lanes guiding vehicles to different destinations:

- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.
- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.
- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.
- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.

Analyze the traffic data from all directions and lanes, and determine if there is a potential conflict between vehicles at the intersection. Respond only with 'Yes' or 'No'.
"""

# Prepare data files
create_finetune_dataset(train_df, 'data/train_finetune.jsonl', system_prompt)
create_finetune_dataset(val_df, 'data/val_finetune.jsonl', system_prompt)

# Verify datasets
verify_dataset('data/train_finetune.jsonl')
verify_dataset('data/val_finetune.jsonl')

# Initialize Together client
client = Together(api_key=api_key)

# Upload files to Together AI
train_upload_response = client.files.upload(file='data/train_finetune.jsonl')
train_file_id = train_upload_response.id
print(f"Training file uploaded. File ID: {train_file_id}")

val_upload_response = client.files.upload(file='data/val_finetune.jsonl')
val_file_id = val_upload_response.id
print(f"Validation file uploaded. File ID: {val_file_id}")

# Fine-tune the model
job_id = fine_tune_model(api_key, train_file_id, val_file_id)

# Monitor completion
monitor_fine_tuning_job(api_key, job_id)

# Save the fine-tuned model ID for later use
# Note: Retrieve the fine-tuned model ID from the Together AI API after completion
