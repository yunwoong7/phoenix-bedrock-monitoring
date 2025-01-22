from phoenix.otel import register
import os
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from functools import wraps

load_dotenv()

PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME")
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT")

def setup_phoenix():
    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    
    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
        verbose=False
    )
    
    return tracer_provider

def trace_function(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            operation_name = name if name else func.__name__
            
            with tracer.start_as_current_span(operation_name) as span:
                try:
                    # Add relevant attributes to the span
                    span.set_attribute("function.name", operation_name)
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Mark the span as successful
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    # Mark the span as failed and record the error
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise
                
        return wrapper
    return decorator