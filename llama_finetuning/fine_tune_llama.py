# llama_finetuning/fine_tune_llama.py

"""
LLAMA Fine-Tuning Module

This module handles the fine-tuning of the LLAMA model for classifying traffic conflicts using the Together AI API.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import time
from together import Together


def fine_tune_model(api_key, train_file_id, val_file_id, model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Reference'):
    """
    Initiates the fine-tuning process of the LLAMA model.

    Args:
        api_key (str): Together AI API key.
        train_file_id (str): File ID of the training data uploaded to Together AI.
        val_file_id (str): File ID of the validation data uploaded to Together AI.
        model_name (str): The base model to fine-tune.

    Returns:
        str: The ID of the fine-tuning job.
    """
    client = Together(api_key=api_key)

    response = client.fine_tuning.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=model_name,
        n_epochs=3,
        batch_size=16,
        learning_rate=3e-5,
        n_evals=5
    )

    job_id = response.id
    print(f"Fine-tuning job started with Job ID: {job_id}")
    return job_id


def monitor_fine_tuning_job(api_key, job_id):
    """
    Monitors the fine-tuning job until completion.

    Args:
        api_key (str): Together AI API key.
        job_id (str): The ID of the fine-tuning job.
    """
    client = Together(api_key=api_key)

    while True:
        job_status = client.fine_tuning.retrieve(job_id)
        status = job_status.status
        print(f"Job Status: {status}")
        if status in ['completed', 'failed', 'cancelled']:
            break
        time.sleep(60)  # Wait for 60 seconds before checking again

    print(f"Fine-tuning job {job_id} has completed with status: {status}")
