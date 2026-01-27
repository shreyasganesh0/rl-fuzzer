# [sysrel]
import socket
import struct
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import csv
import time

SOCKET_PATH = "/tmp/fuzz_rl.sock"
METRICS_FILE = "rl_metrics.csv"
ACTION_SIZE = 10
STATE_SIZE = 1
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3

STRUCT_FMT = "IIIQ" 
PACKET_SIZE = struct.calcsize(STRUCT_FMT)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995 

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            return torch.argmax(self.model(torch.FloatTensor(state).to(self.device))).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE: return 0
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)

        q = self.model(s).gather(1, a)
        next_q = self.model(ns).max(1)[0].unsqueeze(1)
        target = r + (GAMMA * next_q)
        
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

def main():
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)

    # Init CSV
    with open(METRICS_FILE, 'w') as f:
        f.write("step,reward,loss,epsilon,coverage,crashes\n")

    print(f"[*] Brain Logic Ready on {SOCKET_PATH}")
    agent = Agent()
    conn, _ = server.accept()
    
    prev_cov = 0
    prev_crash = 0
    step = 0
    last_state = np.array([0.0])
    last_action = 0

    try:
        while True:
            data = conn.recv(PACKET_SIZE)
            if not data or len(data) < PACKET_SIZE: break
            
            _, cov, crash, execs = struct.unpack(STRUCT_FMT, data)
            state = np.array([float(cov) / 1000.0]) # Use coverage as state for this test

            # Reward Function
            d_cov = cov - prev_cov
            d_crash = crash - prev_crash
            reward = (d_cov * 100) + (d_crash * 10000) - 0.1
            
            # Train
            loss = 0
            if step > 0:
                agent.memory.append((last_state, last_action, reward, state))
                loss = agent.train()

            # Act
            action = agent.act(state)
            conn.send(struct.pack("i", action))

            # Log Metrics every 100 steps
            if step % 100 == 0:
                with open(METRICS_FILE, 'a') as f:
                    f.write(f"{step},{reward:.2f},{loss:.4f},{agent.epsilon:.4f},{cov},{crash}\n")
                print(f"Step {step}: Cov {cov} | Eps {agent.epsilon:.2f} | Act {action}")

            last_state = state
            last_action = action
            prev_cov = cov
            prev_crash = crash
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        os.remove(SOCKET_PATH)

if __name__ == "__main__":
    main()
