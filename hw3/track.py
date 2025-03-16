import csv
import time
import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from config import Config

class Tracker:
    def __init__(self):
        self.folder = "results/"
        self.model_path = os.path.join(self.folder, "model.pth")
        self.history_path = os.path.join(self.folder, "training_history.csv")

        self.rewards_history = []
        self.rps_history = []

        self.start_time = dict()
        self.end_time = dict()

        os.makedirs(self.folder, exist_ok=True)  # Ensure results folder exists        

        # Load existing history if available
        self.episode_loaded = 0
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    self.rewards_history.append(float(row[1]))  # Cumulative reward
                    self.rps_history.append(float(row[2]))      # RPS
            self.episode_loaded = len(self.rewards_history)

    def start_timer(self, worker_id):
        self.start_time[worker_id] = time.time()

    def end_timer(self, worker_id):
        self.end_time[worker_id] = time.time()

    def add_cumulative_reward(self, episode_steps, cumulative_reward, worker_id):
        """Stores cumulative reward and rewards per step in history."""
        self.rewards_history.append(cumulative_reward)
        self.rps_history.append(cumulative_reward / max(episode_steps, 1))
        episode = len(self.rewards_history)

        with open(self.history_path, "a", newline="") as file:
            writer = csv.writer(file)
            if os.stat(self.history_path).st_size == 0:
                writer.writerow(["Episode", "Worker", "Cumulative Reward", "RPS"])

            writer.writerow([episode, worker_id, self.rewards_history[-1], self.rps_history[-1]])

        avg_reward = np.mean(self.rewards_history[-100:])  # Moving average of last 100 episodes
        elapsed_time = self.end_time[worker_id] - self.start_time[worker_id]

        print(f"Episode={episode}, "
              f"Worker={worker_id}, "
              f"Reward={self.rewards_history[-1]:.4f}, "
              f"AvgReward={avg_reward:.4f}, RPS={self.rps_history[-1]:.4f}, "
              f"Time={elapsed_time:.2f}s")

    def load_model(self, model):
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            print(f"Model loaded from {self.model_path}")
        return model
    
    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print(f"Model saved at {self.model_path}")

    def plot_results(self):
        """Plots all recorded rewards and rewards per step (RPS)."""
        if not self.rewards_history:
            print("No training data to plot.")
            return

        window_size_reward = 400
        window_size_rps = 1200

        plt.figure(figsize=(24, 8))

        # Convert to pandas Series for rolling average
        rewards_series = pd.Series(self.rewards_history)
        rps_series = pd.Series(self.rps_history)

        # Compute rolling averages
        rewards_smoothed = rewards_series.rolling(window=window_size_reward, min_periods=1).mean()
        rps_smoothed = rps_series.rolling(window=window_size_rps, min_periods=1).mean()

        # Cumulative Reward Plot
        plt.subplot(1, 2, 1)
        plt.plot(rewards_series, alpha=0.3, label="Raw Cumulative Reward", color="gray")
        plt.plot(rewards_smoothed, label=f"Smoothed (window={window_size_reward})", color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Training Progress - Reward")
        plt.ylim(-0.25, 3)
        plt.legend()

        # Rewards Per Step (RPS) Plot
        plt.subplot(1, 2, 2)
        plt.plot(rps_series, alpha=0.3, label="Raw RPS", color="gray")
        plt.plot(rps_smoothed, label=f"Smoothed (window={window_size_rps})", color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards Per Step")
        plt.title("Training Progress - RPS")
        plt.ylim(-0.01, 0.2)
        plt.legend()

        plt.savefig("training_progress.png")
        plt.show()