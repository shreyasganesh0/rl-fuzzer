import socket
import struct
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

SOCKET_PATH = "/tmp/fuzz_rl.sock"
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-3
MEMORY_SIZE = 10000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-Greedy Strategy
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # State is (1, input_dim)
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*batch)

        state_t = torch.FloatTensor(np.array(batch_state)).to(self.device)
        action_t = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(np.array(batch_next_state)).to(self.device)

        # Compute Q(s, a)
        current_q = self.policy_net(state_t).gather(1, action_t)

        # Compute max Q(s', a') for target
        with torch.no_grad():
            next_q = self.policy_net(next_state_t).max(1)[0].unsqueeze(1)
            target_q = reward_t + (GAMMA * next_q)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def start_server():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    
    # Initialize Agent
    # State: Currently just 1 number (total_execs). Ideally a vector.
    # Actions: 3 mutators (0=Wide, 1=Magic, 2=Havoc)
    agent = Agent(state_dim=1, action_dim=3)
    
    print(f"[*] DQN Brain initialized on {agent.device}")
    print(f"[*] Listening on {SOCKET_PATH}...")

    conn, _ = server.accept()
    print("[+] AFL++ CONNECTED!")
    
    last_state = np.array([0.0])
    last_action = 0

    try:
        total_requests = 0
        while True:
            # CHANGED: Recv 8 bytes (4 byte uint + 4 byte float)
            data = conn.recv(8)
            if not data: break
            
            # 1. UNPACK REAL DATA
            # 'I' = unsigned int (state), 'f' = float (reward)
            val, reward = struct.unpack('<If', data)
            current_state = np.array([float(val)])

            # 2. LOG SIGNIFICANT REWARDS
            if reward > 1.0:
                print(f"[!!!] REWARD RECEIVED: {reward} (Found Path/Crash)")

            # 3. STORE EXPERIENCE & TRAIN
            if total_requests > 0:
                agent.remember(last_state, last_action, reward, current_state)
                agent.train_step()

            # 4. DECIDE NEXT ACTION
            action = agent.select_action(current_state)
            
            conn.send(struct.pack('<I', action))
            
            last_state = current_state
            last_action = action
            total_requests += 1

            if total_requests % 1000 == 0:
                print(f"[*] Step {total_requests}: Eps {agent.steps_done:.2f} | Last R: {reward:.2f}")

    except KeyboardInterrupt:
        print("[!] Stopping...")
    finally:
        conn.close()
        os.remove(SOCKET_PATH)

if __name__ == "__main__":
    start_server()
