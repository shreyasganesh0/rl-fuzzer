/*
   MuoFuzz - RL-Driven Custom Mutator for AFL++
   
   Implements standard AFL++ mutation operators (havoc-style) 
   but controlled by an external RL Agent via Unix Socket.
*/

#include "afl-fuzz.h" // Requires the AFL++ headers (src/include)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#define SOCKET_PATH "/tmp/fuzz_rl.sock"
#define MAX_MUTATED_SIZE (1024 * 1024) 

typedef struct my_mutator {
    afl_state_t *afl;      // Reference to AFL global state
    u8 *mutated_buf;       // Buffer to hold our mutation
    int sock;              // Connection to RL Server
    
    // Metrics tracking for Delta calculation
    u32 prev_cov_count;
    u64 prev_crash_count;
    u64 prev_total_execs;
} my_mutator_t;

typedef struct {
    u32 input_hash;     // Simple state representation
    u32 current_cov;    // Metric: Code Coverage (edge count)
    u32 current_crash;  // Metric: Crash Count
    u64 total_execs;    // Metric: Time/Effort
} rl_packet_t;


// Calculate coverage density from AFL's bitmap
u32 get_coverage_count(afl_state_t *afl) {
    u32 count = 0;
    u32 map_size = afl->fsrv.map_size;
    u8 *map = afl->virgin_bits; 
    
    // Optimization: Count 0xFF (virgin) vs non-0xFF (touched)
    for (u32 i = 0; i < map_size; i++) {
        if (map[i] != 0xFF) count++;
    }
    return count;
}

// Simple hash to give the RL agent a Input State
u32 djb2_hash(u8 *str, size_t len) {
    u32 hash = 5381;
    for (size_t i = 0; i < len; i++)
        hash = ((hash << 5) + hash) + str[i];
    return hash;
}

// Interesting values often trigger bugs (0, MAX_INT, etc.) from AFL++
static const u8  my_interesting_8[]  = {0, 1, 16, 32, 64, 100, 127, 128, 255};
static const u16 my_interesting_16[] = {0, 128, 255, 256, 1024, 32767, 65535};
static const u32 my_interesting_32[] = {0, 1, 32768, 65535, 2147483647, 4294967295};

// Initialize the mutator
my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {
    signal(SIGPIPE, SIG_IGN);
    srand(seed);

    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    if (!data) return NULL;

    data->afl = afl;
    data->mutated_buf = (u8 *)malloc(MAX_MUTATED_SIZE);
    
    data->prev_cov_count = 0;
    data->prev_crash_count = 0;
    data->prev_total_execs = 0;

    // Connect to Python RL Server : [sysrel]
    data->sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(data->sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("[-] [MuoFuzz] Could not connect to RL Server. Running in fallback mode.");
        close(data->sock);
        data->sock = -1;
    } else {
        printf("[+] [MuoFuzz] Connected to RL Brain!\n");
    }

    return data;
}

size_t afl_custom_fuzz(my_mutator_t *data, uint8_t *buf, size_t buf_size, uint8_t **out_buf, uint8_t *add_buf, size_t add_buf_size, size_t max_size) {
    
    // Safety copy of original buffer
    if (buf_size > MAX_MUTATED_SIZE) buf_size = MAX_MUTATED_SIZE;
    memcpy(data->mutated_buf, buf, buf_size);
    
    int action = 0; // Default: 0 (Random/Havoc)

    // --- STEP 1: COMMUNICATE WITH RL --- : [sysrel]
    if (data->sock != -1) {
        rl_packet_t pkt;
        pkt.input_hash = djb2_hash(buf, buf_size);
        pkt.current_cov = get_coverage_count(data->afl);
        pkt.current_crash = data->afl->saved_crashes;
        pkt.total_execs = data->afl->fsrv.total_execs;

        // Send State
        if (send(data->sock, &pkt, sizeof(pkt), 0) == sizeof(pkt)) {
            // Receive Action (0-9)
            if (recv(data->sock, &action, sizeof(int), 0) != sizeof(int)) {
                // If recv fails, assume server died
                close(data->sock);
                data->sock = -1;
            }
        } else {
            close(data->sock);
            data->sock = -1;
        }
    }

    // --- STEP 2: EXECUTE CHOSEN MUTATION --- : [sysrel]
    size_t pos = 0;
    int r = 0;

    switch (action) {
        case 1: // BIT FLIP
            if (buf_size > 0) {
                pos = rand() % buf_size;
                data->mutated_buf[pos] ^= (1 << (rand() % 8));
            }
            break;

        case 2: // BYTE FLIP (XOR 0xFF)
            if (buf_size > 0) {
                pos = rand() % buf_size;
                data->mutated_buf[pos] ^= 0xFF;
            }
            break;

        case 3: // ARITHMETIC ADD (+1 to +35)
            if (buf_size > 0) {
                pos = rand() % buf_size;
                data->mutated_buf[pos] += (1 + (rand() % 35));
            }
            break;

        case 4: // ARITHMETIC SUB (-1 to -35)
            if (buf_size > 0) {
                pos = rand() % buf_size;
                data->mutated_buf[pos] -= (1 + (rand() % 35));
            }
            break;

        case 5: // INTERESTING 8-BIT
            if (buf_size > 0) {
                pos = rand() % buf_size;
                data->mutated_buf[pos] = my_interesting_8[rand() % sizeof(my_interesting_8)];
            }
            break;

        case 6: // INTERESTING 16-BIT (Little Endian)
            if (buf_size >= 2) {
                pos = rand() % (buf_size - 1);
                u16 val = my_interesting_16[rand() % sizeof(my_interesting_16)];
                *(u16*)(data->mutated_buf + pos) = val;
            }
            break;

        case 7: // INTERESTING 32-BIT (Little Endian)
            if (buf_size >= 4) {
                pos = rand() % (buf_size - 3);
                u32 val = my_interesting_32[rand() % sizeof(my_interesting_32)];
                *(u32*)(data->mutated_buf + pos) = val;
            }
            break;

        case 8: // DELETE BYTES (Block Deletion)
            if (buf_size > 2) {
                size_t del_len = 1 + (rand() % (buf_size / 2));
                pos = rand() % (buf_size - del_len);
                // Move memory to cover the hole
                memmove(data->mutated_buf + pos, data->mutated_buf + pos + del_len, buf_size - pos - del_len);
                buf_size -= del_len;
            }
            break;
        
        case 9: // CLONE/INSERT BYTES (Block Duplication)
             if (buf_size + 16 < max_size) {
                 size_t clone_len = 1 + (rand() % 16);
                 if (buf_size + clone_len > MAX_MUTATED_SIZE) clone_len = MAX_MUTATED_SIZE - buf_size;
                 
                 pos = rand() % buf_size;
                 // Make space
                 memmove(data->mutated_buf + pos + clone_len, data->mutated_buf + pos, buf_size - pos);
                 // Fill space with random junk or part of self
                 for(size_t i=0; i<clone_len; i++) data->mutated_buf[pos+i] = rand() % 256;
                 buf_size += clone_len;
             }
             break;

        case 0: // HAVOC / RANDOM (Fallback)
        default:
            // Apply 3 random mutations from the list above
            for(int i=0; i<3; i++) {
                int rnd_act = 1 + (rand() % 9);
                if (buf_size > 0) data->mutated_buf[rand() % buf_size] ^= (rand() % 255 + 1);
            }
            break;
    }

    *out_buf = data->mutated_buf;
    return buf_size;
}

void afl_custom_deinit(my_mutator_t *data) {
    if (data->sock != -1) close(data->sock);
    free(data->mutated_buf);
    free(data);
}
