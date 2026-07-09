import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Please add your Hugging Face token to the .env file."
    )

# Initialize Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def ask_ai(context, question):
    """
    Generate an answer using the retrieved context.
    """

    prompt = f"""
You are an expert Healthcare AI Assistant.

Answer ONLY using the information provided in the context.

If the answer cannot be found in the context, reply exactly:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional healthcare document assistant. "
                        "Answer only from the supplied context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=300,
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
