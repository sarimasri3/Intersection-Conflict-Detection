# llama_finetuning/evaluation.py

"""
Evaluation Module for Fine-Tuned LLAMA Model

This module contains functions to evaluate the fine-tuned LLAMA model on test data.

Author: Your Name
Date: YYYY-MM-DD
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from .together_utils import llama32


def evaluate_model(test_df, prompt):
    """
    Evaluates the fine-tuned LLAMA model on the test dataset.

    Args:
        test_df (pd.DataFrame): Test dataset.
        prompt (str): The system prompt.

    Returns:
        tuple: Final accuracy, confusion matrix, classification report.
    """
    actual_conflicts = []
    predicted_conflicts = []
    count = 1
    correct_predictions = 0
    scenario_total_count = len(test_df)

    # Loop through the dataset to gather predictions and actual values
    for index, row in test_df.iterrows():
        scenario_string = row['scenario']  # 'scenario' column holds the JSON string
        actual_conflict = row['is_conflict'].strip().lower()  # 'yes' or 'no'

        # Detect conflict using LLAMA
        predicted_conflict = detect_conflicts_llama(scenario_string, prompt).lower()  # 'yes' or 'no'

        # Check if prediction is correct and update the count
        if predicted_conflict == actual_conflict:
            correct_predictions += 1

        # Calculate ongoing accuracy
        ongoing_accuracy = correct_predictions / count

        # Print progress along with ongoing accuracy
        print(f"Scenario {count}/{scenario_total_count}, Actual Conflict: {actual_conflict}, Predicted Conflict: {predicted_conflict}, Ongoing Accuracy: {ongoing_accuracy * 100:.2f}%")

        # Increment scenario count
        count += 1

        # Store results for comparison
        actual_conflicts.append(actual_conflict)
        predicted_conflicts.append(predicted_conflict)

    # Generate the confusion matrix
    cm = confusion_matrix(actual_conflicts, predicted_conflicts, labels=['yes', 'no'])

    # Generate the classification report
    report = classification_report(actual_conflicts, predicted_conflicts, labels=['yes', 'no'], target_names=['Conflict', 'No Conflict'])

    # Calculate final accuracy
    final_accuracy = correct_predictions / len(test_df)

    # Print the final results
    print(f"\nFinal Model Accuracy: {final_accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Conflict', 'No Conflict'], yticklabels=['Conflict', 'No Conflict'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return final_accuracy, cm, report


def detect_conflicts_llama(scenario_string, prompt):
    """
    Uses the LLAMA model to detect conflicts in a given traffic scenario.

    Args:
        scenario_string (str): JSON string of the vehicle scenario.
        prompt (str): The system prompt.

    Returns:
        str: 'yes' or 'no' indicating whether there is a conflict.
    """
    # Convert the scenario into a human-readable format
    readable_scenario = parse_scenario_to_string(scenario_string)

    # Set up the system and user messages to pass to LLAMA
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": f"Analyze the following scenario and determine if there is a conflict (Respond only with 'Yes' or 'No'):\n{readable_scenario}"
        }
    ]

    # Get the response from LLAMA
    response = llama32(messages, model_size=11)

    # Extract the model's output (Yes/No)
    answer = response.strip()
    return answer
