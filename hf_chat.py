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

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def ask_ai(context, question):
    prompt = f"""
You are an intelligent Healthcare AI Assistant.

Answer the user's question only using the provided context.
If the answer is not available in the context, reply:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="HuggingFaceTB/SmolLM3-3B",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error while generating response:\n{str(e)}"