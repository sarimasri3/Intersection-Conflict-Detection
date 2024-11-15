# run_llama_evaluation.py

"""
Script to Evaluate Fine-Tuned LLAMA Model

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import pandas as pd
from llama_finetuning.prepare_data import parse_scenario_to_string
from llama_finetuning.evaluation import evaluate_model
from llama_finetuning.together_utils import load_env

# Set your Together AI API key
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("Please set your Together AI API key as an environment variable 'TOGETHER_API_KEY'.")

# Load test dataset
test_df = pd.read_csv('data/test_set.csv')

# Define system prompt
system_prompt = """
You are an Urban Intersection Traffic Conflict Detector, responsible for monitoring a four-way intersection with traffic coming from the north, east, south, and west. Each direction has two lanes guiding vehicles to different destinations:

- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.
- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.
- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.
- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.

Analyze the traffic data from all directions and lanes, and determine if there is a potential conflict between vehicles at the intersection. Respond only with 'Yes' or 'No'.
"""

# Evaluate the model
final_accuracy, cm, report = evaluate_model(test_df, system_prompt)
