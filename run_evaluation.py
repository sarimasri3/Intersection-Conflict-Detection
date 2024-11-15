# run_evaluation.py

from gpt_finetuning.prepare_data import prepare_test_data_for_gpt
from gpt_finetuning.evaluation import predict_and_evaluate, generate_evaluation_report
import openai
import pandas as pd
import os

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'.")

# Load test dataset
test_df = pd.read_csv('data/test_set.csv')

# Load fine-tuned model ID
with open('fine_tuned_model_id.txt', 'r') as f:
    fine_tuned_model_id = f.read().strip()

# Define system instruction
system_instruction = """
You are an Urban Intersection Traffic Conflict Detector, responsible for monitoring a four-way intersection with traffic coming from the north, east, south, and west. Each direction has two lanes guiding vehicles to different destinations:

- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.
- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.
- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.
- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.

Analyze the traffic data from all directions and lanes, and determine if there is a potential conflict between vehicles at the intersection. Respond only with 'yes' or 'no'.
"""

# Prepare test data
test_data = prepare_test_data_for_gpt(test_df, system_instruction)

# Evaluate the model
y_true, y_pred = predict_and_evaluate(test_data, fine_tuned_model_id, openai_api_key)

# Generate evaluation report
generate_evaluation_report(y_true, y_pred)
