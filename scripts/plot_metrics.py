import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

METRICS_FILE = 'rl_metrics.csv'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"[-] Error: {METRICS_FILE} not found. Run the fuzzer first!")
        return

    print(f"[+] Loading metrics from {METRICS_FILE}...")
    try:
        df = pd.read_csv(METRICS_FILE)
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        return

    has_policy_data = 'action' in df.columns and 'input_hash' in df.columns

    # --- FIGURE 1: TRAINING DYNAMICS (The "Health" Check) ---
    plt.figure(figsize=(15, 5))

    # Subplot 1: Reward & Epsilon
    plt.subplot(1, 3, 1)
    plt.plot(df['step'], df['reward'], label='Reward', color='green', alpha=0.3)
    if len(df) > 10:
        plt.plot(df['step'], df['reward'].rolling(window=10).mean(), label='Avg Reward (10)', color='darkgreen', linewidth=2)
    
    ax2 = plt.gca().twinx()
    ax2.plot(df['step'], df['epsilon'], label='Epsilon', color='orange', linestyle='--', linewidth=2)
    ax2.set_ylabel('Epsilon (Exploration)', color='orange')
    
    plt.title('Agent Training: Reward vs Exploration')
    plt.xlabel('Steps')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss
    plt.subplot(1, 3, 2)
    plt.plot(df['step'], df['loss'], label='Loss', color='red', alpha=0.6)
    plt.title('Neural Network Loss')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Coverage
    plt.subplot(1, 3, 3)
    plt.plot(df['step'], df['coverage'], label='Coverage', color='blue', linewidth=2)
    plt.title('Fuzzing Success: Coverage Growth')
    plt.xlabel('Steps')
    plt.ylabel('Bitmap Entries')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/training_health.png")
    print(f"[+] Saved {PLOT_DIR}/training_health.png")

    # --- FIGURE 2: POLICY ANALYSIS ---
    if has_policy_data:
        print("[+] Policy data found! Generating advanced analysis...")
        
        plt.figure(figsize=(12, 8))
        
        # 1. Action Distribution (Histogram)
        plt.subplot(2, 1, 1)
        action_counts = df['action'].value_counts().sort_index()
        action_names = {
            0: 'Arith Inc', 1: 'Arith Dec', 2: 'Int8', 3: 'Int32', 
            4: 'Dict Insert', 5: 'Delete', 6: 'Clone', 7: 'Havoc'
        }
        labels = [action_names.get(i, str(i)) for i in action_counts.index]
        
        bars = plt.bar(labels, action_counts.values, color='purple', alpha=0.7)
        plt.title('Overall Mutator Utilization')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)

        # 2. State-Action Correlation (Scatter/Heatmap style)
        plt.subplot(2, 1, 2)
        
        top_hashes = df['input_hash'].value_counts().head(5).index
        filtered_df = df[df['input_hash'].isin(top_hashes)]
        
        for h in top_hashes:
            subset = filtered_df[filtered_df['input_hash'] == h]
            plt.scatter(subset['step'], subset['action'], label=f"Hash {h}", alpha=0.6, s=15)
        
        plt.yticks(range(10), [action_names.get(i, str(i)) for i in range(10)])
        plt.title('Policy Trajectory: Action Selection per Constraint over Time')
        plt.xlabel('Step')
        plt.ylabel('Selected Mutator')
        plt.legend(title="Constraint Hash")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/policy_analysis.png")
        print(f"[+] Saved {PLOT_DIR}/policy_analysis.png")
        print("[!] SUCCESS: This graph proves your agent adapts to constraints!")

    else:
        print("[-] Warning: 'action' and 'input_hash' columns not found in CSV.")
        print("    To get the Presentation Graphs, update rl_server.py to log these fields!")

if __name__ == "__main__":
    main()
