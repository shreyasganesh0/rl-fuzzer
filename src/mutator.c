/*
   [sysrel]
*/

#include "afl-fuzz.h" 

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#define SOCKET_PATH "/tmp/fuzz_rl.sock"
#define MAX_MUTATED_SIZE (1024 * 1024) 

static const int32_t MY_INTERESTING_8[] = { -128, -1, 0, 1, 16, 32, 64, 100, 127 };
static const int16_t MY_INTERESTING_16[] = { -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767 };
static const int32_t MY_INTERESTING_32[] = { -2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2147483647 };

typedef struct my_mutator {
    afl_state_t *afl;      // Access to global AFL state
    u8 *mutated_buf;       // Buffer for our mutations
    int sock;              // Connection to RL Server
    
    // Metrics
    u32 prev_cov_count;
    u64 prev_crash_count;
} my_mutator_t;

typedef struct {
    u32 input_hash;     
    u32 current_cov;    
    u32 current_crash;  
    u64 total_execs;    
} rl_packet_t;

// --- DJB2 HASH ---
u32 djb2_hash(const u8 *str, size_t len) {
    u32 hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

// --- INIT ---
my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {
    srand(seed);
    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    data->afl = afl; 
    data->mutated_buf = malloc(MAX_MUTATED_SIZE);

    data->sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(data->sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("[-] Failed to connect to RL Server");
    } else {
        printf("[+] Mutator connected to RL Brain.\n");
    }

    return data;
}

// --- FUZZ ---
size_t afl_custom_fuzz(my_mutator_t *data, uint8_t *buf, size_t buf_size, 
                       uint8_t **out_buf, uint8_t *add_buf, 
                       size_t add_buf_size, size_t max_size) {
    
    // 1. Send State to RL
    u32 current_hash = djb2_hash(buf, buf_size);
    
    rl_packet_t packet;
    packet.input_hash = current_hash;
    packet.current_cov = data->afl->total_bitmap_size;
    packet.current_crash = data->afl->total_crashes;
    
    // FIXED: total_execs resides in the 'fsrv' struct in modern AFL++
    packet.total_execs = data->afl->fsrv.total_execs; 

    int action = 0; 
    if (data->sock > 0) {
        if (send(data->sock, &packet, sizeof(packet), 0) < 0) {
             // Handle disconnect silently
        } else {
             recv(data->sock, &action, sizeof(int), 0);
        }
    }

    // 2. Setup Mutation Buffer
    size_t mut_len = buf_size;
    if (mut_len > MAX_MUTATED_SIZE) mut_len = MAX_MUTATED_SIZE;
    memcpy(data->mutated_buf, buf, mut_len);

    // 3. Execute Strategy
    int pos = 0;
    
    switch(action) {
        case 0: // ARITHMETIC INC
            if (mut_len > 0) {
                pos = rand() % mut_len;
                data->mutated_buf[pos]++;
            }
            break;

        case 1: // ARITHMETIC DEC
            if (mut_len > 0) {
                pos = rand() % mut_len;
                data->mutated_buf[pos]--;
            }
            break;

        case 2: // INTERESTING VAL 8
            if (mut_len > 0) {
                pos = rand() % mut_len;
                // FIXED: Using renamed array
                int8_t val = MY_INTERESTING_8[rand() % (sizeof(MY_INTERESTING_8)/sizeof(int32_t))];
                data->mutated_buf[pos] = (u8)val;
            }
            break;
            
        case 3: // INTERESTING VAL 32
             if (mut_len >= 4) {
                pos = rand() % (mut_len - 3);
                // FIXED: Using renamed array
                int32_t val = MY_INTERESTING_32[rand() % (sizeof(MY_INTERESTING_32)/sizeof(int32_t))];
                *(int32_t*)(data->mutated_buf + pos) = val;
             }
             break;

        case 4: // DICTIONARY INSERT
             if (data->afl->extras_cnt > 0 || data->afl->a_extras_cnt > 0) {
                 int use_extra = 1;
                 if (data->afl->a_extras_cnt > 0 && (rand() % 2 == 0)) use_extra = 0;
                 if (data->afl->extras_cnt == 0) use_extra = 0;

                 u8 *token_data = NULL;
                 u32 token_len = 0;

                 if (use_extra) {
                     u32 idx = rand() % data->afl->extras_cnt;
                     token_data = data->afl->extras[idx].data;
                     token_len = data->afl->extras[idx].len;
                 } else {
                     u32 idx = rand() % data->afl->a_extras_cnt;
                     token_data = data->afl->a_extras[idx].data;
                     token_len = data->afl->a_extras[idx].len;
                 }

                 if (token_data && token_len > 0 && mut_len + token_len <= max_size) {
                     if (rand() % 2 == 0) { // INSERT
                         pos = rand() % (mut_len + 1);
                         memmove(data->mutated_buf + pos + token_len, data->mutated_buf + pos, mut_len - pos);
                         memcpy(data->mutated_buf + pos, token_data, token_len);
                         mut_len += token_len;
                     } else { // OVERWRITE
                         if (mut_len >= token_len) {
                             pos = rand() % (mut_len - token_len + 1);
                             memcpy(data->mutated_buf + pos, token_data, token_len);
                         }
                     }
                 }
             } else {
                 goto HAVOC_LABEL; 
             }
             break;

        case 5: // DELETE BYTES
            if (mut_len > 1) {
                u32 del_len = 1 + (rand() % (mut_len - 1));
                if (del_len >= mut_len) del_len = mut_len - 1;
                
                pos = rand() % (mut_len - del_len);
                memmove(data->mutated_buf + pos, data->mutated_buf + pos + del_len, mut_len - pos - del_len);
                mut_len -= del_len;
            }
            break;

        default: 
        HAVOC_LABEL:
            for(int i=0; i < (1 + rand() % 4); i++) {
                if (mut_len == 0) break;
                int method = rand() % 3;
                pos = rand() % mut_len;
                
                if (method == 0) data->mutated_buf[pos] ^= (1 << (rand() % 8));
                else if (method == 1) data->mutated_buf[pos] = rand() % 256;
                else data->mutated_buf[pos] = ~data->mutated_buf[pos];
            }
            break;
    }

    *out_buf = data->mutated_buf;
    return mut_len;
}

void afl_custom_deinit(my_mutator_t *data) {
    if (data->sock > 0) close(data->sock);
    free(data->mutated_buf);
    free(data);
}
