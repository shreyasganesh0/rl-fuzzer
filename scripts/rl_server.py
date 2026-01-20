import socket
import os

# Define the socket file location
SOCKET_PATH = "/tmp/fuzz_rl.sock"

# Clean up old socket if it exists
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

print("[*] RL Server starting... Waiting for AFL++")
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)

conn, _ = server.accept()
print("[+] AFL++ Connected!")

try:
    while True:
        # Receive State from C (e.g., 4 bytes for an ID)
        data = conn.recv(4)
        if not data: break
        
        state_id = int.from_bytes(data, "little")
        # print(f"[*] Received State: {state_id}")
        
        action = state_id % 3
        
        # Send Action back to C
        conn.send(action.to_bytes(4, "little"))
finally:
    conn.close()
    os.remove(SOCKET_PATH)
