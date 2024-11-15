# gpt_finetuning/fine_tune_gpt.py

"""
GPT Fine-Tuning Module

This module handles the fine-tuning of the GPT model for classifying traffic conflicts.

Author: Your Name
Date: YYYY-MM-DD
"""

import openai
import time


def fine_tune_model(openai_api_key, train_file_id, val_file_id, base_model="gpt-3.5-turbo"):
    """
    Initiates the fine-tuning process of the GPT model.

    Parameters:
    - openai_api_key: Your OpenAI API key.
    - train_file_id: File ID of the training data uploaded to OpenAI.
    - val_file_id: File ID of the validation data uploaded to OpenAI.
    - base_model: The base model to fine-tune.

    Returns:
    - fine_tune_id: The ID of the fine-tuning job.
    """
    openai.api_key = openai_api_key

    fine_tune_response = openai.FineTune.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=base_model
    )

    fine_tune_id = fine_tune_response.id
    print(f"Fine-tuning started. Fine-tune job ID: {fine_tune_id}")
    return fine_tune_id


def wait_for_fine_tuning_completion(openai_api_key, fine_tune_id):
    """
    Waits for the fine-tuning job to complete and returns the fine-tuned model ID.

    Parameters:
    - openai_api_key: Your OpenAI API key.
    - fine_tune_id: The ID of the fine-tuning job.

    Returns:
    - fine_tuned_model_id: The ID of the fine-tuned model.
    """
    openai.api_key = openai_api_key

    while True:
        fine_tune_status = openai.FineTune.retrieve(fine_tune_id)
        status = fine_tune_status.status
        if status == 'succeeded':
            fine_tuned_model_id = fine_tune_status.fine_tuned_model
            print(f"Fine-tuning succeeded. Fine-tuned Model ID: {fine_tuned_model_id}")
            return fine_tuned_model_id
        elif status == 'failed':
            print(f"Fine-tuning failed. Error: {fine_tune_status}")
            return None
        else:
            print(f"Fine-tuning status: {status}. Waiting for completion...")
            time.sleep(60)  # Wait for 1 minute before checking again
