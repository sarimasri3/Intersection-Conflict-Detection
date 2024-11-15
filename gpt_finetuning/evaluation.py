# gpt_finetuning/evaluation.py

"""
Evaluation Module for Fine-Tuned GPT Model

This module contains functions to evaluate the fine-tuned GPT model on test data.

Author: Your Name
Date: YYYY-MM-DD
"""

import openai
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def predict_and_evaluate(test_data, fine_tuned_model_id, openai_api_key):
    """
    Predicts and evaluates the fine-tuned GPT model on the test dataset.

    Parameters:
    - test_data: List of dictionaries containing test scenarios in GPT's chat format.
    - fine_tuned_model_id: The ID of the fine-tuned GPT model.
    - openai_api_key: Your OpenAI API key.

    Returns:
    - y_true: List of true labels (from the dataset).
    - y_pred: List of predicted labels (from the model).
    """
    openai.api_key = openai_api_key

    y_true = []
    y_pred = []
    correct_predictions = 0
    count = 0

    # Loop through each test example
    for item in test_data:
        # Retrieve system instructions and user message
        system_message = item['messages'][0]['content']  # System instruction
        user_message = item['messages'][1]['content']    # Traffic scenario
        true_label = item['messages'][2]['content'].strip()  # True label ('yes' or 'no')

        # Get the fine-tuned model's prediction
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id,
            messages=[
                {"role": "system", "content": system_message},  # Custom instruction
                {"role": "user", "content": user_message}       # Scenario prompt
            ],
            max_tokens=5,
            temperature=0.0
        )

        # Extract the predicted label from the model's response
        predicted_label = response.choices[0].message.content.strip().lower()

        # Append true and predicted labels to the lists
        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Check if the prediction matches the true label
        if predicted_label == true_label:
            correct_predictions += 1

        # Increment the count
        count += 1

        # Calculate ongoing accuracy
        ongoing_accuracy = correct_predictions / count

        # Print the current prediction and ongoing accuracy
        print(f"Scenario {count}/{len(test_data)}, True: {true_label}, Predicted: {predicted_label}, Ongoing Accuracy: {ongoing_accuracy * 100:.2f}%")

    return y_true, y_pred


def generate_evaluation_report(y_true, y_pred):
    """
    Generates and displays evaluation metrics including classification report and confusion matrix.

    Parameters:
    - y_true: List of true labels.
    - y_pred: List of predicted labels.
    """
    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Conflict: Yes', 'Conflict: No']))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['yes', 'no'])
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accuracy score
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
