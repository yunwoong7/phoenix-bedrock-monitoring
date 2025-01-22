from phoenix.otel import register
import os
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from functools import wraps

load_dotenv()

# ==== Load environment variables ====
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME")
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT")

# ==== Setup Phoenix Tracing ====
def setup_phoenix():
    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    
    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
        verbose=False
    )
    
    return tracer_provider
