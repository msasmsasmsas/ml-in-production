import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import grpc
import concurrent.futures
import json
import os
from typing import Dict, List, Tuple, Any, Optional

# For gRPC communication - assumed to be in a proto file
try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError:
    print("Warning: gRPC proto modules not found. gRPC benchmarking will be unavailable.")

class ModelServerBenchmark:
    """Benchmarks model server performance by components"""

    def __init__(self, rest_url: str = None, grpc_server: str = None):
        self.rest_url = rest_url
        self.grpc_server = grpc_server
        self.results = {}

    def benchmark_rest(self, image_path: str, num_requests: int = 100, concurrency: int = 1) -> Dict:
        """Benchmark REST API with component timing"""
        results = {
            'total_latency': [],
            'network_latency': [],
            'server_processing': [],
            'data_preprocessing': [],
            'model_inference': [],
            'postprocessing': []
        }

        def single_request(image_path):
            # Measure client-side preprocessing time
            preprocess_start = time.time()
            # In real implementation, you would read and process the image here
            with open(image_path, 'rb') as f:
                image_data = f.read()
            preprocess_time = time.time() - preprocess_start

            # Measure network and server processing time
            headers = {'X-Benchmark': 'true'}
            request_start = time.time()
            response = requests.post(
                f"{self.rest_url}/predict", 
                files={'image': image_data},
                headers=headers
            )
            request_end = time.time()

            if response.status_code != 200:
                return None

            # Extract component timings from response headers
            component_times = json.loads(response.headers.get('X-Timing-Info', '{}'))

            return {
                'total_latency': request_end - request_start,
                'network_latency': (request_end - request_start) - component_times.get('server_processing', 0),
                'server_processing': component_times.get('server_processing', 0),
                'data_preprocessing': component_times.get('preprocessing', 0),
                'model_inference': component_times.get('inference', 0),
                'postprocessing': component_times.get('postprocessing', 0),
                'client_preprocessing': preprocess_time
            }

        # Execute requests (sequential or concurrent)
        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_request, image_path) for _ in range(num_requests)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        for key in results:
                            results[key].append(result.get(key, 0))
        else:
            for _ in range(num_requests):
                result = single_request(image_path)
                if result:
                    for key in results:
                        results[key].append(result.get(key, 0))

        # Convert to numpy arrays for statistics
        for key in results:
            results[key] = np.array(results[key])

        self.results['rest'] = results
        return self._calculate_statistics(results)

    def benchmark_grpc(self, image_path: str, num_requests: int = 100, concurrency: int = 1) -> Dict:
        """Benchmark gRPC API with component timing"""
        results = {
            'total_latency': [],
            'network_latency': [],
            'server_processing': [],
            'data_preprocessing': [],
            'model_inference': [],
            'postprocessing': []
        }

        def single_request(image_path):
            # Measure client-side preprocessing time
            preprocess_start = time.time()
            # In real implementation, you would read and process the image here
            with open(image_path, 'rb') as f:
                image_data = f.read()
            preprocess_time = time.time() - preprocess_start

            # Create gRPC channel and stub
            channel = grpc.insecure_channel(self.grpc_server)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)

            # Create request with timing metadata
            request = inference_pb2.PredictRequest(
                image_data=image_data,
                enable_timing=True
            )

            # Measure network and server processing time
            request_start = time.time()
            response = stub.Predict(request)
            request_end = time.time()

            # Extract component timings from response metadata
            component_times = {}
            for timing in response.timing_info:
                component_times[timing.component] = timing.duration_ms / 1000  # Convert to seconds

            return {
                'total_latency': request_end - request_start,
                'network_latency': (request_end - request_start) - component_times.get('server_processing', 0),
                'server_processing': component_times.get('server_processing', 0),
                'data_preprocessing': component_times.get('preprocessing', 0),
                'model_inference': component_times.get('inference', 0),
                'postprocessing': component_times.get('postprocessing', 0),
                'client_preprocessing': preprocess_time
            }

        # Execute requests (sequential or concurrent)
        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_request, image_path) for _ in range(num_requests)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    for key in results:
                        results[key].append(result.get(key, 0))
        else:
            for _ in range(num_requests):
                result = single_request(image_path)
                for key in results:
                    results[key].append(result.get(key, 0))

        # Convert to numpy arrays for statistics
        for key in results:
            results[key] = np.array(results[key])

        self.results['grpc'] = results
        return self._calculate_statistics(results)

    def _calculate_statistics(self, results: Dict[str, np.ndarray]) -> Dict:
        """Calculate statistical metrics for benchmark results"""
        stats = {}
        for component, values in results.items():
            if len(values) == 0:
                continue

            stats[component] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99)),
                'std_dev': float(np.std(values))
            }
        return stats

    def plot_component_comparison(self, output_path: str = 'component_benchmark.png'):
        """Generate comparative plots of component benchmarks"""
        if not self.results:
            print("No benchmark results to plot")
            return

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Server Component Benchmarking', fontsize=16)

        # Flatten the axes for easier iteration
        axs = axs.flatten()

        # Plot 1: Component time breakdown (stacked bar chart)
        ax = axs[0]
        components = ['data_preprocessing', 'model_inference', 'postprocessing', 'network_latency']
        protocols = list(self.results.keys())

        bottom = np.zeros(len(protocols))
        for component in components:
            means = [np.mean(self.results[protocol][component]) * 1000 for protocol in protocols]  # Convert to ms
            ax.bar(protocols, means, bottom=bottom, label=component)
            bottom += means

        ax.set_title('Time Breakdown by Component')
        ax.set_ylabel('Time (ms)')
        ax.legend()

        # Plot 2: Comparative bar chart for total latency
        ax = axs[1]
        means = [np.mean(self.results[protocol]['total_latency']) * 1000 for protocol in protocols]  # Convert to ms
        p95s = [np.percentile(self.results[protocol]['total_latency'], 95) * 1000 for protocol in protocols]

        x = np.arange(len(protocols))
        width = 0.35
        ax.bar(x - width/2, means, width, label='Mean')
        ax.bar(x + width/2, p95s, width, label='95th Percentile')

        ax.set_title('Total Latency Comparison')
        ax.set_ylabel('Latency (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(protocols)
        ax.legend()

        # Plot 3: Component percentage breakdown (stacked percentage bar)
        ax = axs[2]
        for i, protocol in enumerate(protocols):
            total = np.mean(self.results[protocol]['total_latency'])
            percentages = []
            labels = []

            for component in components:
                mean_time = np.mean(self.results[protocol][component])
                percentage = (mean_time / total) * 100
                percentages.append(percentage)
                labels.append(f"{component}\n({percentage:.1f}%)")

            ax.pie(percentages, labels=labels, autopct='%1.1f%%',
                   startangle=90, radius=0.8 + (i * 0.2))

        ax.set_title('Component Time Percentage')

        # Plot 4: Cumulative distribution function of latencies
        ax = axs[3]
        for protocol in protocols:
            latencies = sorted(self.results[protocol]['total_latency'] * 1000)  # Convert to ms
            y = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(latencies, y, label=protocol)

        ax.set_title('Latency CDF')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()
        print(f"Component benchmark plot saved to {output_path}")

    def save_results(self, output_path: str = 'component_benchmark_results.json'):
        """Save benchmark results to JSON file"""
        if not self.results:
            print("No benchmark results to save")
            return

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for protocol, components in self.results.items():
            serializable_results[protocol] = {}
            for component, values in components.items():
                serializable_results[protocol][component] = values.tolist()

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Benchmark results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark model server by components')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--rest-url', help='URL of REST API server')
    parser.add_argument('--grpc-server', help='Address of gRPC server')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--output-json', default='component_benchmark_results.json', help='Output JSON file path')
    parser.add_argument('--output-plot', default='component_benchmark.png', help='Output plot file path')
    args = parser.parse_args()

    if not args.rest_url and not args.grpc_server:
        parser.error("At least one of --rest-url or --grpc-server must be provided")

    benchmark = ModelServerBenchmark(args.rest_url, args.grpc_server)

    if args.rest_url:
        print(f"Benchmarking REST API at {args.rest_url}...")
        rest_stats = benchmark.benchmark_rest(args.image, args.requests, args.concurrency)
        print("\nREST API Component Statistics:")
        for component, stats in rest_stats.items():
            print(f"  {component}:")
            for metric, value in stats.items():
                print(f"    {metric}: {value * 1000:.2f} ms")  # Convert to ms for display

    if args.grpc_server:
        print(f"\nBenchmarking gRPC server at {args.grpc_server}...")
        grpc_stats = benchmark.benchmark_grpc(args.image, args.requests, args.concurrency)
        print("\ngRPC Server Component Statistics:")
        for component, stats in grpc_stats.items():
            print(f"  {component}:")
            for metric, value in stats.items():
                print(f"    {metric}: {value * 1000:.2f} ms")  # Convert to ms for display

    # Generate visualizations and save results
    benchmark.plot_component_comparison(args.output_plot)
    benchmark.save_results(args.output_json)

if __name__ == '__main__':
    main()
