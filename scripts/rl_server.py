# [sysrel]
# State Vector: [Coverage, CMP_Type, Bit_Width, Is_Const, Depth, PREV_ACTION]

import socket
import struct
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json

# --- CONFIGURATION ---
SOCKET_PATH = "/tmp/fuzz_rl.sock"
METRICS_FILE = "rl_metrics.csv"
CONSTRAINTS_FILE = "constraints.json"

# State: 
# 1. Global Coverage (Normalized)
# 2-5. Static Features (Type, Width, Const, Depth)
# 6. Previous Action (Normalized)
STATE_SIZE = 6 

ACTION_SIZE = 10 
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3

STRUCT_FMT = "IIIQ" 
PACKET_SIZE = struct.calcsize(STRUCT_FMT)

# --- DQN MODEL ---
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
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return np.argmax(q_values.cpu().data.numpy())

    def train(self):
        if len(self.memory) < BATCH_SIZE: return 0
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        loss_val = 0
        for state, action, reward, next_state in minibatch:
            state_t = torch.FloatTensor(state).to(self.device)
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            
            target = reward + GAMMA * torch.max(self.model(next_state_t))
            prediction = self.model(state_t)[action]
            
            loss = (target - prediction) ** 2
            loss_val += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_val / BATCH_SIZE

def load_constraints():
    if not os.path.exists(CONSTRAINTS_FILE):
        return {}
    with open(CONSTRAINTS_FILE, 'r') as f:
        return json.load(f)

def main():
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    
    print(f"[+] Static MuoFuzz Brain listening on {SOCKET_PATH}...")
    
    agent = Agent()
    constraints = load_constraints()
    
    with open(METRICS_FILE, 'w') as f:
        f.write("step,reward,loss,epsilon,coverage,crashes,action,input_hash\n")

    conn, _ = server.accept()
    
    prev_cov = 0
    prev_crash = 0
    step = 0
    
    last_action_taken = 0 
    
    last_state = np.zeros(STATE_SIZE)

    try:
        while True:
            data = conn.recv(PACKET_SIZE)
            if not data or len(data) < PACKET_SIZE: break
            
            input_hash, cov, crash, execs = struct.unpack(STRUCT_FMT, data)
            
            # --- BUILD STATE ---
            cov_feature = float(cov) / 1000.0
            static_features = constraints.get(str(input_hash), [0.0, 0.0, 0.0, 0.0])
            
            # NORMALIZE PREV ACTION: Scale 0-9 to 0.0-1.0
            action_feature = float(last_action_taken) / float(ACTION_SIZE)
            
            # State Vector now includes history
            state = np.array([cov_feature] + static_features + [action_feature])

            # --- REWARD ---
            d_cov = cov - prev_cov
            d_crash = crash - prev_crash
            reward = (d_cov * 100.0) + (d_crash * 10000.0) - 0.1
            
            if d_cov > 0 and sum(static_features) > 0:
                reward += 20.0 

            # --- TRAIN ---
            loss = 0
            if step > 0:
                agent.memory.append((last_state, last_action_taken, reward, state))
                loss = agent.train()

            # --- ACT ---
            action = agent.act(state)
            conn.send(struct.pack("i", action))

            if step % 100 == 0:
                with open(METRICS_FILE, 'a') as f:
                    f.write(f"{step},{reward:.2f},{loss:.4f},{agent.epsilon:.4f},{cov},{crash},{action},{input_hash}\n")

            # Update Loop
            last_state = state
            last_action_taken = action 
            prev_cov = cov
            prev_crash = crash
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)

if __name__ == "__main__":
    main()
