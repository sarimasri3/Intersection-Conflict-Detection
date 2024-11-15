# run_fine_tuning.py

from gpt_finetuning.prepare_data import prepare_chat_jsonl_file
from gpt_finetuning.fine_tune_gpt import fine_tune_model, wait_for_fine_tuning_completion
import openai
import pandas as pd
import os

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'.")

# Load datasets
train_df = pd.read_csv('data/train_set.csv')
val_df = pd.read_csv('data/val_set.csv')

# Define system instruction
system_instruction = """
You are an Urban Intersection Traffic Conflict Detector, responsible for monitoring a four-way intersection with traffic coming from the north, east, south, and west. Each direction has two lanes guiding vehicles to different destinations:

- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.
- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.
- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.
- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.

Analyze the traffic data from all directions and lanes, and determine if there is a potential conflict between vehicles at the intersection. Respond only with 'yes' or 'no'.
"""

# Prepare data files
prepare_chat_jsonl_file(train_df, 'data/train_data.jsonl', system_instruction)
prepare_chat_jsonl_file(val_df, 'data/val_data.jsonl', system_instruction)

# Upload files to OpenAI
train_file_response = openai.File.create(
    file=open('data/train_data.jsonl', 'rb'),
    purpose="fine-tune"
)

val_file_response = openai.File.create(
    file=open('data/val_data.jsonl', 'rb'),
    purpose="fine-tune"
)

# Get file IDs
train_file_id = train_file_response.id
val_file_id = val_file_response.id

# Fine-tune the model
fine_tune_id = fine_tune_model(openai_api_key, train_file_id, val_file_id)

# Wait for completion
fine_tuned_model_id = wait_for_fine_tuning_completion(openai_api_key, fine_tune_id)

# Save the fine-tuned model ID for later use
with open('fine_tuned_model_id.txt', 'w') as f:
    f.write(fine_tuned_model_id)
