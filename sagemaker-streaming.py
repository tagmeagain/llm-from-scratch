import boto3
import json
import time
from typing import Generator, Dict, Any

class StreamingSageMakerClient:
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime')
    
    def stream_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream the response from the SageMaker endpoint using invoke_endpoint_with_response_stream
        """
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
                            yield data['text']
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error processing chunk: {e}")
                    continue
    
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
    
    # Example prompts
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "What are the key differences between Python and JavaScript?"
    ]
    
    # Test streaming response
    print("Testing Streaming Response:")
    print("=" * 50)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Response:")
        start_time = time.time()
        
        try:
            # Stream the response
            for text in client.stream_response(prompt):
                print(text, end="", flush=True)
            
            end_time = time.time()
            print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"\nError during streaming: {e}")
        
        print("-" * 50)
    
    # Test complete response
    print("\nTesting Complete Response:")
    print("=" * 50)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Response:")
        start_time = time.time()
        
        try:
            # Get complete response
            response = client.get_complete_response(prompt)
            print(response)
            
            end_time = time.time()
            print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"\nError during complete response: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 
