#include "afl-fuzz.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>

#ifndef _FIXED_CHAR
#define _FIXED_CHAR 0x41
#define SOCKET_PATH "/tmp/fuzz_rl.sock"
#endif

typedef struct my_mutator {
  afl_state_t *afl;
  u8 *fuzz_buf;
  int sock;
} my_mutator_t;

my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {

    signal(SIGPIPE, SIG_IGN);

    srand(seed);
    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    if (!data) return NULL;

    data->fuzz_buf = (u8 *)malloc(MAX_FILE);
    if (!data->fuzz_buf) {
        free(data);
        return NULL;
    }

    data->afl = afl;

    // Setup socket
    data->sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(data->sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      perror("[-] Failed to connect to RL Server. Running in offline mode.");
      // Don't crash, just mark socket as invalid so we can fallback
      close(data->sock);
      data->sock = -1; 
    } else {
        printf("[+] Connected to RL Brain!\n");
    }

    return data;
}

size_t afl_custom_fuzz(my_mutator_t *data, uint8_t *buf, size_t buf_size,
                       u8 **out_buf, uint8_t *add_buf,
                       size_t add_buf_size,
                       size_t max_size) {

    if (buf_size == 0) {
        *out_buf = buf;
        return 0;
    }

    u32 action_id = 2; // Default to Havoc

    if (data->sock != -1) {
        u32 state_id = 0; // Placeholder for now
        
        if (write(data->sock, &state_id, sizeof(state_id)) != sizeof(state_id)) {
             data->sock = -1;
        } else {
            // Read response
            if (read(data->sock, &action_id, sizeof(action_id)) != sizeof(action_id)) {
                data->sock = -1;
            }
        }
    }

    // Copy original buffer
    memcpy(data->fuzz_buf, buf, buf_size);

    switch(action_id) {
        case 0:// wide range mutator
            data->fuzz_buf[0] = rand() % 255;
            break;
        case 1:// magic byte mutator
            if (buf_size >= 4) memcpy(data->fuzz_buf, "BAAD", 4);
            break;
        default:// havoc mutator
            data->fuzz_buf[rand() % buf_size] ^= 0xFF;
    }

    *out_buf = data->fuzz_buf;
    return buf_size;
}

void afl_custom_deinit(my_mutator_t *data) {
    if (data->sock != -1) close(data->sock);
    free(data->fuzz_buf);
    free(data);
}
