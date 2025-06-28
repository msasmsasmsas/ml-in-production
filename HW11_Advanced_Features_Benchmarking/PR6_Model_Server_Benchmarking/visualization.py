import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

class BenchmarkVisualizer:
    """Visualization tools for model server component benchmarks"""

    def __init__(self, results_file: Optional[str] = None):
        self.results = {}
        if results_file:
            self.load_results(results_file)

    def load_results(self, results_file: str):
        """Load benchmark results from JSON file"""
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Convert lists to numpy arrays
        self.results = {}
        for protocol, components in data.items():
            self.results[protocol] = {}
            for component, values in components.items():
                self.results[protocol][component] = np.array(values)

    def plot_component_latencies(self, output_path: str = 'component_latencies.png'):
        """Plot bar chart of component latencies"""
        if not self.results:
            print("No results to plot")
            return

        # Extract mean latencies for each component and protocol
        data = []
        for protocol in self.results.keys():
            for component in ['data_preprocessing', 'model_inference', 'postprocessing', 'network_latency']:
                if component in self.results[protocol]:
                    mean_latency = np.mean(self.results[protocol][component]) * 1000  # Convert to ms
                    data.append({
                        'Protocol': protocol,
                        'Component': component,
                        'Latency (ms)': mean_latency
                    })

        df = pd.DataFrame(data)

        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Component', y='Latency (ms)', hue='Protocol', data=df)
        plt.title('Component Latency Comparison')
        plt.ylabel('Mean Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Component latencies plot saved to {output_path}")

    def plot_latency_distributions(self, output_path: str = 'latency_distributions.png'):
        """Plot distributions of total latencies"""
        if not self.results:
            print("No results to plot")
            return

        protocols = list(self.results.keys())
        plt.figure(figsize=(12, 6))

        for protocol in protocols:
            if 'total_latency' in self.results[protocol]:
                latencies = self.results[protocol]['total_latency'] * 1000  # Convert to ms
                sns.kdeplot(latencies, label=protocol)

        plt.title('Distribution of Total Latencies')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Latency distributions plot saved to {output_path}")

    def plot_component_percentages(self, output_path: str = 'component_percentages.png'):
        """Plot stacked bar chart of component time percentages"""
        if not self.results:
            print("No results to plot")
            return

        components = ['data_preprocessing', 'model_inference', 'postprocessing', 'network_latency']
        protocols = list(self.results.keys())

        # Prepare data
        df_data = []
        for protocol in protocols:
            total_mean = np.mean(self.results[protocol]['total_latency'])
            for component in components:
                if component in self.results[protocol]:
                    component_mean = np.mean(self.results[protocol][component])
                    percentage = (component_mean / total_mean) * 100
                    df_data.append({
                        'Protocol': protocol,
                        'Component': component,
                        'Percentage': percentage
                    })

        df = pd.DataFrame(df_data)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Pivot data for stacked bars
        pivot_df = df.pivot(index='Protocol', columns='Component', values='Percentage')
        pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())

        plt.title('Component Time Percentage Breakdown')
        plt.xlabel('Protocol')
        plt.ylabel('Percentage of Total Time (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Component')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Component percentages plot saved to {output_path}")

    def plot_comparative_dashboard(self, output_path: str = 'benchmark_dashboard.png'):
        """Create a comprehensive dashboard of benchmark results"""
        if not self.results:
            print("No results to plot")
            return

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Server Component Benchmarking Dashboard', fontsize=16)

        # Flatten the axes for easier iteration
        axs = axs.flatten()

        # Plot 1: Component time breakdown (stacked bar chart)
        ax = axs[0]
        components = ['data_preprocessing', 'model_inference', 'postprocessing', 'network_latency']
        protocols = list(self.results.keys())

        bottom = np.zeros(len(protocols))
        for component in components:
            means = [np.mean(self.results[protocol].get(component, [0])) * 1000 for protocol in protocols]  # Convert to ms
            ax.bar(protocols, means, bottom=bottom, label=component)
            bottom += means

        ax.set_title('Time Breakdown by Component')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

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
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot 3: Component percentage breakdown
        ax = axs[2]
        for i, protocol in enumerate(protocols):
            # Create pie chart for each protocol
            ax_pie = fig.add_axes([0.1 + i * 0.4, 0.1, 0.3, 0.3])

            total = np.mean(self.results[protocol]['total_latency'])
            percentages = []
            labels = []

            for component in components:
                if component in self.results[protocol]:
                    mean_time = np.mean(self.results[protocol][component])
                    percentage = (mean_time / total) * 100
                    percentages.append(percentage)
                    labels.append(f"{component}\n({percentage:.1f}%)")

            ax_pie.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
            ax_pie.set_title(f"{protocol} Time Breakdown")

        # Plot 4: Latency CDF
        ax = axs[3]
        for protocol in protocols:
            latencies = sorted(self.results[protocol]['total_latency'] * 1000)  # Convert to ms
            y = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(latencies, y, label=protocol)

        ax.set_title('Latency CDF')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()
        print(f"Benchmark dashboard saved to {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results', required=True, help='Path to benchmark results JSON file')
    parser.add_argument('--output-dir', default='.', help='Directory to save visualization outputs')
    args = parser.parse_args()

    visualizer = BenchmarkVisualizer(args.results)

    # Generate various visualizations
    visualizer.plot_component_latencies(f"{args.output_dir}/component_latencies.png")
    visualizer.plot_latency_distributions(f"{args.output_dir}/latency_distributions.png")
    visualizer.plot_component_percentages(f"{args.output_dir}/component_percentages.png")
    visualizer.plot_comparative_dashboard(f"{args.output_dir}/benchmark_dashboard.png")

if __name__ == '__main__':
    main()
