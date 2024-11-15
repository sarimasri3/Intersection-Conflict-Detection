# gpt_finetuning/prepare_data.py

"""
Data Preparation Module for GPT Fine-Tuning

This module contains functions to prepare training, validation, and test data
for fine-tuning the GPT model to classify traffic conflicts.

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

        vehicle_description = (f"Vehicle {vehicle_id} is in lane {lane}, moving {direction} at a speed of "
                               f"{speed:.2f} km/h, and is {distance:.2f} meters away from the intersection, "
                               f"heading towards {destination}.")
        scenario_description.append(vehicle_description)

    readable_string = " ".join(scenario_description)
    return readable_string


def prepare_chat_jsonl_file(df, file_path, system_instruction):
    """
    Converts the DataFrame to a JSONL file format for GPT fine-tuning in chat format.

    Parameters:
    - df: DataFrame containing the dataset.
    - file_path: Path to the output JSONL file.
    - system_instruction: Custom instruction for the system message.
    """
    with open(file_path, 'w') as jsonl_file:
        for index, row in df.iterrows():
            # Convert the scenario to a human-readable string
            scenario_string = parse_scenario_to_string(row['scenario'])

            # Prepare the conversation with system, user, and assistant messages
            conversation = {
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Analyze the following scenario and determine if there is a conflict (Respond only with 'yes' or 'no'): {scenario_string}"},
                    {"role": "assistant", "content": row['is_conflict'].strip().lower()}  # Ensure it's either 'yes' or 'no'
                ]
            }

            # Write the JSON object as a new line in the JSONL file
            jsonl_file.write(json.dumps(conversation) + '\n')


def prepare_test_data_for_gpt(df, system_instruction):
    """
    Prepares test data for GPT in chat format with system, user, and assistant roles.

    Parameters:
    - df: DataFrame containing the test data.
    - system_instruction: Instruction for the system message.

    Returns:
    - List of dictionaries where each dictionary is a chat conversation for GPT.
    """
    test_data = []

    for _, row in df.iterrows():
        # Convert the scenario to a human-readable string
        scenario_string = parse_scenario_to_string(row['scenario'])

        # Create a message conversation with system, user, and assistant
        conversation = {
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Analyze the following scenario and determine if there is a conflict (Respond only with 'yes' or 'no'): {scenario_string}"},
                {"role": "assistant", "content": row['is_conflict'].strip().lower()}  # True label: 'yes' or 'no'
            ]
        }

        test_data.append(conversation)

    return test_data
