import socket
import struct
import os
import random

SOCKET_PATH = "/tmp/fuzz_rl.sock"

def start_server():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    # 2. SETUP: Create the Unix Domain Socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    
    print(f"[*] Python Brain listening on {SOCKET_PATH}...")
    print("[*] Waiting for AFL++ to connect...")

    # 3. ACCEPT: Wait for the C mutator to connect
    conn, _ = server.accept()
    print("[+] AFL++ CONNECTED! Handshake successful.")

    try:
        total_requests = 0
        while True:
            # 4. RECEIVE STATE: Read 4 bytes (uint32) from C
            data = conn.recv(4)
            if not data:
                break # Connection closed
            
            # Unpack binary data into an integer
            state_id = struct.unpack('<I', data)[0]
            
            # For now, random logic to test the pipe.
            action_id = random.randint(0, 2) 

            # 5. SEND ACTION: Pack integer back to 4 bytes
            conn.send(struct.pack('<I', action_id))
            
            total_requests += 1
            if total_requests % 1000 == 0:
                print(f"[*] Processed {total_requests} mutations. Last State: {state_id}")

    except KeyboardInterrupt:
        print("[!] Stopping server...")
    finally:
        conn.close()
        os.remove(SOCKET_PATH)

if __name__ == "__main__":
    start_server()
