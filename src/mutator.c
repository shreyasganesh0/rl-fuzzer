#include "afl-fuzz.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>

#define SOCKET_PATH "/tmp/fuzz_rl.sock"

// sysrel
typedef struct my_mutator {
  afl_state_t *afl;
  u8 *fuzz_buf;
  int sock;
  // Track previous stats to calculate New finds
  u64 prev_queued_items;
  u64 prev_saved_crashes;
} my_mutator_t;

my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {
    signal(SIGPIPE, SIG_IGN);
    srand(seed);
    
    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    if (!data) {
        perror("[-] Failed to allocate mutator data");
        return NULL;
    }

    data->fuzz_buf = (u8 *)malloc(MAX_FILE);
    if (!data->fuzz_buf) {
        perror("[-] Failed to allocate fuzz buffer");
        free(data);
        return NULL;
    }

    data->afl = afl;
    data->prev_queued_items = afl->queued_items;
    data->prev_saved_crashes = afl->saved_crashes; 

    // sysrel
    data->sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (data->sock == -1) {
        perror("[-] Failed to create socket");
        free(data->fuzz_buf);
        free(data);
        return NULL;
    }

    // sysrel
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(data->sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      perror("[-] Offline Mode");
      close(data->sock);
      data->sock = -1; 
    } else {
        printf("[+] Connected to RL Brain!\n");
    }
    return data;
}

size_t afl_custom_fuzz(my_mutator_t *data, uint8_t *buf, size_t buf_size,
                       u8 **out_buf, uint8_t *add_buf, size_t add_buf_size, size_t max_size) {

    // Safety checks
    if (!data || !buf || buf_size == 0) { 
        if (out_buf) *out_buf = buf; 
        return 0; 
    }

    // Bounds check
    if (buf_size > MAX_FILE) {
        buf_size = MAX_FILE;
    }

    // sysrel
    u32 action_id = 2; // Default Havoc

    if (data->sock != -1) {
        // --- 1. CALCULATE REAL REWARD ---
        float reward = -0.1f; // Small step cost to encourage speed

        // Check internal AFL counters
        if (data->afl->queued_items > data->prev_queued_items) {
            reward = 10.0f; // BIG REWARD for finding new code path!
            data->prev_queued_items = data->afl->queued_items;
        }
        if (data->afl->saved_crashes > data->prev_saved_crashes) {  // Changed from unique_crashes
            reward = 100.0f; // HUGE REWARD for crashing!
            data->prev_saved_crashes = data->afl->saved_crashes;
        }

        // --- 2. PREPARE PACKET (State + Reward) ---
        u32 state_id = (u32)data->afl->fsrv.total_execs;

        struct {
            u32 state;
            float reward;
        } packet;
        
        packet.state = state_id;
        packet.reward = reward;

        // --- 3. SEND & RECEIVE ---
        ssize_t written = write(data->sock, &packet, sizeof(packet));
        if (written != sizeof(packet)) {
             close(data->sock);
             data->sock = -1;
        } else {
            ssize_t bytes_read = read(data->sock, &action_id, sizeof(action_id));
            if (bytes_read != sizeof(action_id)) {
                close(data->sock);
                data->sock = -1;
            }
        }
    }

    memcpy(data->fuzz_buf, buf, buf_size);

    // sysrel
    // Apply Mutator with bounds checking
    switch(action_id) {
        case 0: // Wide Range
            if (buf_size > 0) {
                data->fuzz_buf[0] = rand() % 256;
            }
            break;
        case 1: // Magic Byte
            if (buf_size >= 4) {
                data->fuzz_buf[0] = 'B';
                data->fuzz_buf[1] = 'A';
                data->fuzz_buf[2] = 'D';
                data->fuzz_buf[3] = '!';
            }
            break;
        default: // Havoc
            if (buf_size > 0) {
                size_t pos = rand() % buf_size;
                data->fuzz_buf[pos] ^= 0xFF;
            }
            break;
    }

    *out_buf = data->fuzz_buf;
    return buf_size;
}

void afl_custom_deinit(my_mutator_t *data) {
    if (!data) return;
    if (data->sock != -1) close(data->sock);
    if (data->fuzz_buf) free(data->fuzz_buf);
    free(data);
}
