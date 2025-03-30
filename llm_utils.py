import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import time  # Import time for potential retries/delays

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Configuration ---
# Choose appropriate models:
# Main model: For complex reasoning, planning, reflection (e.g., Gemini 1.5 Pro)
# Assistant model: For faster, simpler tasks like query generation, initial refinement (e.g., Gemini 1.5 Flash)
MAIN_MODEL_NAME = "gemini-1.5-pro-latest"
ASSISTANT_MODEL_NAME = (
    "gemini-1.5-flash-latest"  # Use Flash for potentially faster/cheaper tasks
)
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- Client Initialization ---
# The client is configured globally via genai.configure()


def get_main_model():
    """Returns an instance of the configured main generative model."""
    try:
        model = genai.GenerativeModel(MAIN_MODEL_NAME)
        print(f"Main generative model '{MAIN_MODEL_NAME}' initialized.")
        return model
    except Exception as e:
        print(f"Error initializing main generative model: {e}")
        raise


def get_assistant_model():
    """Returns an instance of the configured assistant generative model."""
    try:
        model = genai.GenerativeModel(ASSISTANT_MODEL_NAME)
        print(f"Assistant generative model '{ASSISTANT_MODEL_NAME}' initialized.")
        return model
    except Exception as e:
        print(f"Error initializing assistant generative model: {e}")
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
        print(
            "Warning: generate_text called without explicit model, defaulting to assistant model."
        )
        model = get_assistant_model()

    print(f"\n--- Generating Text using {model.model_name} ---")
    print(f"Prompt: {prompt[:100]}...")  # Log truncated prompt
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = model.generate_content(prompt)

            # Check if response has text
            if response.parts:
                print(f"Response: {response.text[:100]}...")  # Log truncated response
                return response.text

            # If no text, check for blocking or other issues
            print("Warning: Received empty response part from LLM.")
            print(f"Full Response: {response}")  # Log full response for debugging

            # Check for prompt feedback (e.g., safety blocks)
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
                # Return None or raise specific error if prompt was blocked
                return f"Error: Prompt blocked due to safety settings - {response.prompt_feedback}"

            # Check finish reason in candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                print(f"Candidate Finish Reason: {candidate.finish_reason}")
                if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    print(f"Candidate Safety Ratings: {candidate.safety_ratings}")
                # Handle specific finish reasons if necessary
                if candidate.finish_reason != "STOP":
                    return f"Error: Generation stopped unexpectedly - Reason: {candidate.finish_reason}"

            # If no text and no clear error, return None after logging
            return None

        except Exception as e:
            print(
                f"Error during text generation (Attempt {retries + 1}/{MAX_RETRIES}): {e}"
            )
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("Max retries reached. Failed to generate text.")
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
        print("Warning: generate_embeddings called with empty list.")
        return []

    print(
        f"Generating embeddings for {len(texts)} text(s) using model '{model_name}' with task type '{task_type}'..."
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # The embed_content function handles batching internally
            result = genai.embed_content(
                model=model_name, content=texts, task_type=task_type
            )
            print(f"Successfully generated {len(result['embedding'])} embeddings.")
            return result["embedding"]
        except Exception as e:
            print(
                f"Error generating embeddings (Attempt {retries + 1}/{MAX_RETRIES}): {e}"
            )
            retries += 1
            if retries < MAX_RETRIES:
                # Check for specific retryable errors if possible (e.g., rate limits)
                # For now, retry on any exception
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("Max retries reached. Failed to generate embeddings.")
                return None  # Return None after max retries

    return None  # Should not be reached, safeguard return


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Get instances of both models for testing
    try:
        main_model_instance = get_main_model()
        assistant_model_instance = get_assistant_model()
    except Exception as e:
        print(f"Failed to initialize models for example usage: {e}")
        exit()

    test_prompt_complex = "Develop a multi-step plan to research the effectiveness of various memory retrieval strategies for autonomous agents."
    print("\n--- Testing Main Model ---")
    generated_text_main = generate_text(test_prompt_complex, model=main_model_instance)

    if generated_text_main:
        print("\n--- Generated Text (Main) ---")
        if generated_text_main.startswith("Error:"):
            print(f"Generation failed: {generated_text_main}")
        else:
            print(generated_text_main)
    else:
        print("\nFailed to generate text with main model after retries.")

    test_prompt_simple = "Generate a short search query for 'latest Mars rover news'."
    print("\n--- Testing Assistant Model ---")
    generated_text_assistant = generate_text(
        test_prompt_simple, model=assistant_model_instance
    )

    if generated_text_assistant:
        print("\n--- Generated Text (Assistant) ---")
        if generated_text_assistant.startswith("Error:"):
            print(f"Generation failed: {generated_text_assistant}")
        else:
            print(generated_text_assistant)
    else:
        print("\nFailed to generate text with assistant model after retries.")

    # Example embedding generation
    test_texts = ["Hello world", "This is a test.", "How does embedding work?"]
    embeddings = generate_embeddings(
        test_texts, task_type="SEMANTIC_SIMILARITY"
    )  # Example task type
    if embeddings:
        print("\n--- Generated Embeddings ---")
        print(f"Number of embeddings: {len(embeddings)}")
        if embeddings:
            print(f"Dimension of first embedding: {len(embeddings[0])}")
            # print(f"First embedding (truncated): {embeddings[0][:10]}...") # Optional: view part of the vector
    else:
        print("\nFailed to generate embeddings after retries.")
