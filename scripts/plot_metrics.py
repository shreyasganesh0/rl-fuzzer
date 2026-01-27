import pandas as pd
import matplotlib.pyplot as plt
import os

METRICS_FILE = 'rl_metrics.csv'
PLOT_FILE = 'plots/rl_training_plot.png'

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"[-] No metrics file found at {METRICS_FILE}. Run the fuzzer first.")
        return

    try:
        data = pd.read_csv(METRICS_FILE)
        
        plt.figure(figsize=(12, 6))

        # Plot 1: Coverage Growth
        plt.subplot(1, 2, 1)
        plt.plot(data['step'], data['coverage'], label='Coverage', color='blue', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Bitmap Entries')
        plt.title('Coverage Growth')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: Reward vs Epsilon
        plt.subplot(1, 2, 2)
        plt.plot(data['step'], data['epsilon'], label='Epsilon (Exploration)', color='orange', linestyle='--')
        plt.plot(data['step'], data['reward'], label='Reward', color='green', alpha=0.3)
        plt.xlabel('Steps')
        plt.title('Agent Logic')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        print(f"[+] Saving plot to {PLOT_FILE}...")
        plt.savefig(PLOT_FILE)
        print(f"[+] Total Crashes Found: {data['crashes'].max()}")

    except Exception as e:
        print(f"[-] Error plotting metrics: {e}")

if __name__ == "__main__":
    main()
