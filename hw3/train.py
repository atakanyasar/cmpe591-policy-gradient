import torch
import numpy as np
from collections import deque
from torch.multiprocessing import Process, Pipe
from homework3 import Hw3Env
from model import REINFORCE
from agent import Agent
from config import Config
from track import Tracker

def collect_trajectory(env_id, config, model, conn):
    """ Worker function that collects a single trajectory and sends it via Pipe. """
    try:
        env = Hw3Env(render_mode="offscreen")
        agent = Agent(model)  # Create agent in worker

        trajectory = []
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0

        while not done:
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            trajectory.append((state, action, reward, next_state, done))
            cumulative_reward += reward
            state = next_state

        conn.send((trajectory, cumulative_reward, None))  # Send data with no error
    except Exception as e:
        conn.send((None, None, str(e)))  # Send error message to parent
    finally:
        conn.close()

def train(config: Config, model: REINFORCE, tracker: Tracker):
    num_workers = config.num_envs
    trajectory_buffer = deque(maxlen=config.batch_size)
    
    parent_conns, child_conns = zip(*[Pipe() for _ in range(num_workers)])
    
    agent = Agent(model)  # Define agent in main process

    try:
        for episode in range(config.num_episodes):
            processes = []  # Reset process list each episode
            
            # Start workers
            for env_id in range(num_workers):
                tracker.start_timer(env_id)
                p = Process(target=collect_trajectory, args=(env_id, config, model, child_conns[env_id]))
                p.start()
                processes.append(p)

            # Collect trajectories and check for errors
            for env_id, conn in enumerate(parent_conns):
                trajectory, cumulative_reward, error = conn.recv()
                tracker.end_timer(env_id)

                if error is not None:
                    raise RuntimeError(f"Child process error in worker {env_id}: {error}")  # Terminate immediately

                trajectory_buffer.append(trajectory)
                tracker.add_cumulative_reward(len(trajectory), cumulative_reward, worker_id=env_id)

            # Train when batch is full
            if len(trajectory_buffer) >= config.batch_size:
                agent.update_model(trajectory_buffer)  
                trajectory_buffer.clear()

            if episode % config.save_interval == 0:
                tracker.save_model(model)

            # Ensure all processes are cleaned up
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

    except Exception as e:
        print(f"\nException occurred: {e}\nTerminating all processes...")

        # Terminate all child processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

        tracker.save_model(model)
        tracker.plot_results()
        print("All processes terminated. Exiting safely.")
        exit(1)

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

if __name__ == "__main__":
    config = Config()
    tracker = Tracker()
    model = REINFORCE().to(config.device)
    tracker.load_model(model)

    try:
        train(config, model, tracker)
    except KeyboardInterrupt:
        print("Training interrupted")
        tracker.save_model(model)
        tracker.plot_results()
