# llama_finetuning/prepare_data.py

"""
Data Preparation Module for LLAMA Fine-Tuning

This module contains functions to prepare training, validation, and test data
for fine-tuning the LLAMA model to classify traffic conflicts.

Author: Your Name
Date: YYYY-MM-DD
"""

import json
import pandas as pd


def parse_scenario_to_string(scenario_string):
    """
    Converts a vehicle scenario JSON string into a readable text description.

    Args:
        scenario_string (str): JSON string of the vehicle scenario.

    Returns:
        str: Formatted text description of the scenario.
    """
    scenario_data = json.loads(scenario_string)
    vehicles = scenario_data.get("vehicles_scenario", [])
    scenario_description = []

    for vehicle in vehicles:
        vehicle_id = vehicle.get("vehicle_id", "Unknown")
        lane = vehicle.get("lane", "Unknown")
        speed = vehicle.get("speed", "Unknown")
        distance = vehicle.get("distance_to_intersection", "Unknown")
        direction = vehicle.get("direction", "Unknown")
        destination = vehicle.get("destination", "Unknown")

        vehicle_description = (
            f"Vehicle {vehicle_id} is in lane {lane}, moving {direction} at a speed of "
            f"{speed:.2f} km/h, and is {distance:.2f} meters away from the intersection, "
            f"heading towards {destination}."
        )
        scenario_description.append(vehicle_description)

    readable_string = " ".join(scenario_description)
    return readable_string


def create_finetune_dataset(df, output_file, system_prompt, model='llama3'):
    """
    Creates a fine-tuning dataset in JSONL format suitable for LLAMA models.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        output_file (str): Path to the output JSONL file.
        system_prompt (str): The system prompt to include.
        model (str): The model version ('llama3' or others).
    """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, row in df.iterrows():
            scenario_string = row['scenario']
            is_conflict = row['is_conflict'].strip()

            readable_scenario = parse_scenario_to_string(scenario_string)

            # Prepare the user input
            user_input = (
                "Analyze the following scenario and determine if there is a conflict "
                f"(Respond only with 'Yes' or 'No'):\n{readable_scenario}"
            )

            # Depending on the model, format accordingly
            text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{is_conflict}<|eot_id|>"""

            # Write to the JSONL file
            json_line = json.dumps({"text": text})
            f_out.write(json_line + '\n')


def verify_dataset(file_path):
    """
    Verifies the dataset by checking for any missing 'text' fields and prints examples.

    Args:
        file_path (str): Path to the JSONL file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data.get('text', '')
            if not text:
                print(f"Line {i} is missing 'text' field.")
            else:
                # Optionally print the first few examples
                if i < 2:
                    print(f"Example {i+1}:\n{text}\n")
    print(f"Verification completed for {file_path}.")
