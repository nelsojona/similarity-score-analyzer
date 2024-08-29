import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configuration settings
CONFIG = {
    "GOOGLE_CLOUD_NLP_API_KEY": os.getenv("GOOGLE_CLOUD_NLP_API_KEY"),
    "MODEL_NAME": "Universal Sentence Encoder",
}
