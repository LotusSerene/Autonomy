import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import time  # Import time for potential retries/delays
import logging

# Load environment variables
load_dotenv()

# Create a logger instance
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # This error is critical, so using raise is appropriate.
    # Logging can be added here if desired, but the raise will stop execution.
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Configuration ---
# Choose appropriate models:
# Main model: For complex reasoning, planning, reflection
# Assistant model: For faster, simpler tasks like query generation, initial refinement
MAIN_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
ASSISTANT_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- Client Initialization ---
# The client is configured globally via genai.configure()


def get_main_model():
    """Returns an instance of the configured main generative model."""
    try:
        model = genai.GenerativeModel(MAIN_MODEL_NAME)
        logger.info(f"Main generative model '{MAIN_MODEL_NAME}' initialized.")
        return model
    except Exception as e:
        logger.exception(f"Error initializing main generative model: {e}")
        raise


def get_assistant_model():
    """Returns an instance of the configured assistant generative model."""
    try:
        model = genai.GenerativeModel(ASSISTANT_MODEL_NAME)
        logger.info(f"Assistant generative model '{ASSISTANT_MODEL_NAME}' initialized.")
        return model
    except Exception as e:
        logger.exception(f"Error initializing assistant generative model: {e}")
        raise


# --- LLM Operations ---

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def generate_text(prompt: str, model=None) -> Optional[str]:
    """
    Generates text using the provided Google AI model instance with basic error checking.
    If no model is provided, it defaults to the ASSISTANT model for safety/speed.
    """
    if model is None:
        # Default to the assistant model if none is specified explicitly
        logger.warning(
            "generate_text called without explicit model, defaulting to assistant model."
        )
        model = get_assistant_model()

    logger.info(f"--- Generating Text using {model.model_name} ---")
    logger.info(f"Prompt: {prompt[:100]}...")  # Log truncated prompt
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = model.generate_content(prompt)

            # Check if response has text
            if response.parts:
                logger.info(f"Response: {response.text[:100]}...")  # Log truncated response
                return response.text

            # If no text, check for blocking or other issues
            logger.warning("Received empty response part from LLM.")
            logger.warning(f"Full Response: {response}")  # Log full response for debugging

            # Check for prompt feedback (e.g., safety blocks)
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                logger.error(f"Prompt Feedback: {response.prompt_feedback}")
                # Return None or raise specific error if prompt was blocked
                return f"Error: Prompt blocked due to safety settings - {response.prompt_feedback}"

            # Check finish reason in candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                logger.warning(f"Candidate Finish Reason: {candidate.finish_reason}")
                if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    logger.warning(f"Candidate Safety Ratings: {candidate.safety_ratings}")
                # Handle specific finish reasons if necessary
                if candidate.finish_reason != "STOP":
                    return f"Error: Generation stopped unexpectedly - Reason: {candidate.finish_reason}"

            # If no text and no clear error, return None after logging
            return None

        except Exception as e:
            logger.error(
                f"Error during text generation (Attempt {retries + 1}/{MAX_RETRIES}): {e}", exc_info=True
            )
            retries += 1
            if retries < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Max retries reached. Failed to generate text.")
                return None  # Return None after max retries

    return None  # Should not be reached if loop logic is correct, but as a safeguard


# Implement embedding generation using the Google AI embedding model
def generate_embeddings(
    texts: List[str],
    task_type="RETRIEVAL_DOCUMENT",
    model_name=f"models/{EMBEDDING_MODEL_NAME}",
) -> Optional[List[List[float]]]:
    """
    Generates embeddings for a list of texts using the configured Google AI model.
    Includes basic retry logic.

    Args:
        texts: A list of strings to embed.
        task_type: The task type for the embedding (e.g., "RETRIEVAL_DOCUMENT",
                   "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY").
        model_name: The specific embedding model identifier.

    Returns:
        A list of embedding vectors (lists of floats), or None if an error occurs
        after retries.
    """
    if not texts:
        logger.warning("generate_embeddings called with empty list.")
        return []

    logger.info(
        f"Generating embeddings for {len(texts)} text(s) using model '{model_name}' with task type '{task_type}'..."
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # The embed_content function handles batching internally
            result = genai.embed_content(
                model=model_name, content=texts, task_type=task_type
            )
            logger.info(f"Successfully generated {len(result['embedding'])} embeddings.")
            return result["embedding"]
        except Exception as e:
            logger.error(
                f"Error generating embeddings (Attempt {retries + 1}/{MAX_RETRIES}): {e}", exc_info=True
            )
            retries += 1
            if retries < MAX_RETRIES:
                # Check for specific retryable errors if possible (e.g., rate limits)
                # For now, retry on any exception
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Max retries reached. Failed to generate embeddings.")
                return None  # Return None after max retries

    return None  # Should not be reached, safeguard return


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Get instances of both models for testing
    try:
        main_model_instance = get_main_model()
        assistant_model_instance = get_assistant_model()
    except Exception as e:
        logger.exception(f"Failed to initialize models for example usage: {e}")
        exit()

    test_prompt_complex = "Develop a multi-step plan to research the effectiveness of various memory retrieval strategies for autonomous agents."
    logger.info("\n--- Testing Main Model ---")
    generated_text_main = generate_text(test_prompt_complex, model=main_model_instance)

    if generated_text_main:
        logger.info("\n--- Generated Text (Main) ---")
        if generated_text_main.startswith("Error:"):
            logger.error(f"Generation failed: {generated_text_main}")
        else:
            logger.info(generated_text_main)
    else:
        logger.error("\nFailed to generate text with main model after retries.")

    test_prompt_simple = "Generate a short search query for 'latest Mars rover news'."
    logger.info("\n--- Testing Assistant Model ---")
    generated_text_assistant = generate_text(
        test_prompt_simple, model=assistant_model_instance
    )

    if generated_text_assistant:
        logger.info("\n--- Generated Text (Assistant) ---")
        if generated_text_assistant.startswith("Error:"):
            logger.error(f"Generation failed: {generated_text_assistant}")
        else:
            logger.info(generated_text_assistant)
    else:
        logger.error("\nFailed to generate text with assistant model after retries.")

    # Example embedding generation
    test_texts = ["Hello world", "This is a test.", "How does embedding work?"]
    embeddings = generate_embeddings(
        test_texts, task_type="SEMANTIC_SIMILARITY"
    )  # Example task type
    if embeddings:
        logger.info("\n--- Generated Embeddings ---")
        logger.info(f"Number of embeddings: {len(embeddings)}")
        if embeddings: # This check is redundant due to the outer `if embeddings:`
            logger.info(f"Dimension of first embedding: {len(embeddings[0])}")
            # logger.debug(f"First embedding (truncated): {embeddings[0][:10]}...") # Optional: view part of the vector
    else:
        logger.error("\nFailed to generate embeddings after retries.")
