import boto3
import json
import time
from typing import Generator, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class ResponseMetrics:
    time_to_first_token: float
    tokens_per_second: float
    tokens_per_minute: float
    total_time: float
    total_tokens: int
    prompt_length: int
    prompt_tokens: int

class MetricsTracker:
    def __init__(self):
        self.metrics_history: List[ResponseMetrics] = []
    
    def add_metrics(self, metrics: ResponseMetrics):
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        
        return {
            'avg_time_to_first_token': sum(m.time_to_first_token for m in self.metrics_history) / len(self.metrics_history),
            'avg_tokens_per_second': sum(m.tokens_per_second for m in self.metrics_history) / len(self.metrics_history),
            'avg_tokens_per_minute': sum(m.tokens_per_minute for m in self.metrics_history) / len(self.metrics_history),
            'avg_total_time': sum(m.total_time for m in self.metrics_history) / len(self.metrics_history),
            'avg_total_tokens': sum(m.total_tokens for m in self.metrics_history) / len(self.metrics_history),
            'avg_prompt_tokens': sum(m.prompt_tokens for m in self.metrics_history) / len(self.metrics_history)
        }
    
    def save_to_csv(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'streaming_metrics_{timestamp}.csv'
        
        df = pd.DataFrame([vars(m) for m in self.metrics_history])
        df.to_csv(filename, index=False)
        print(f"\nMetrics saved to {filename}")

class StreamingSageMakerClient:
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime')
        self.metrics_tracker = MetricsTracker()
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting (approximate)"""
        return len(text.split())
    
    def stream_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream the response from the SageMaker endpoint using invoke_endpoint_with_response_stream
        """
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        prompt_tokens = self.count_tokens(prompt)
        
        response = self.runtime.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature
            })
        )
        
        # Process the streaming response
        for event in response['EventStream']:
            if 'PayloadPart' in event:
                try:
                    chunk = event['PayloadPart']['Bytes'].decode('utf-8')
                    if chunk.startswith('data: '):
                        data = json.loads(chunk[6:])  # Remove 'data: ' prefix
                        if 'text' in data:
                            if first_token_time is None:
                                first_token_time = time.time()
                            text = data['text']
                            total_tokens += self.count_tokens(text)
                            yield text
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error processing chunk: {e}")
                    continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate and store metrics
        metrics = ResponseMetrics(
            time_to_first_token=first_token_time - start_time if first_token_time else 0,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            tokens_per_minute=(total_tokens / total_time) * 60 if total_time > 0 else 0,
            total_time=total_time,
            total_tokens=total_tokens,
            prompt_length=len(prompt),
            prompt_tokens=prompt_tokens
        )
        self.metrics_tracker.add_metrics(metrics)
    
    def get_complete_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Get the complete response at once (non-streaming)
        """
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature
            })
        )
        
        return response['Body'].read().decode('utf-8')

def main():
    # Initialize the client
    client = StreamingSageMakerClient("your-endpoint-name")  # Replace with your endpoint name
    
    # Benchmark prompts covering different types of tasks
    benchmark_prompts = [
        # Short responses
        "What is machine learning?",
        "Explain quantum computing in one sentence.",
        "Define artificial intelligence.",
        
        # Medium responses
        "Write a short story about a robot learning to paint.",
        "What are the key differences between Python and JavaScript?",
        "Describe the process of photosynthesis in detail.",
        
        # Long responses
        "Write a comprehensive guide about implementing a neural network from scratch.",
        "Explain the history and development of quantum computing, including major breakthroughs and current challenges.",
        "Write a detailed analysis of climate change, its causes, effects, and potential solutions.",
        
        # Technical explanations
        "Explain how a transformer model works in natural language processing.",
        "Describe the process of training a deep learning model.",
        "What are the key components of a modern computer architecture?",
        
        # Creative writing
        "Write a poem about artificial intelligence.",
        "Create a short story about a future where humans and AI coexist.",
        "Write a dialogue between a human and an advanced AI system."
    ]
    
    print("Starting Benchmark Tests")
    print("=" * 50)
    
    # Test streaming response with metrics
    for i, prompt in enumerate(benchmark_prompts, 1):
        print(f"\nTest {i}/{len(benchmark_prompts)}")
        print(f"Prompt: {prompt}")
        print("Response:")
        
        try:
            # Stream the response
            for text in client.stream_response(prompt):
                print(text, end="", flush=True)
            
            # Get metrics for this response
            metrics = client.metrics_tracker.metrics_history[-1]
            print(f"\n\nMetrics for this response:")
            print(f"Time to first token: {metrics.time_to_first_token:.2f} seconds")
            print(f"Tokens per second: {metrics.tokens_per_second:.2f}")
            print(f"Tokens per minute: {metrics.tokens_per_minute:.2f}")
            print(f"Total time: {metrics.total_time:.2f} seconds")
            print(f"Total tokens: {metrics.total_tokens}")
            print(f"Prompt tokens: {metrics.prompt_tokens}")
            
        except Exception as e:
            print(f"\nError during streaming: {e}")
        
        print("-" * 50)
    
    # Print and save overall metrics
    print("\nOverall Metrics Summary:")
    print("=" * 50)
    avg_metrics = client.metrics_tracker.get_average_metrics()
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Save metrics to CSV
    client.metrics_tracker.save_to_csv()

if __name__ == "__main__":
    main() 
