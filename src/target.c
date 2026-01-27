// [sysrel]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void crash() {
    char *ptr = NULL;
    *ptr = 'A'; 
}

int main(int argc, char **argv) {
    char buf[1024];
    int len;

    // Read from stdin (AFL default)
    len = read(0, buf, sizeof(buf));
    if (len < 0) return 0;

    // --- CHECK 1: Magic Bytes (Needs Magic Injection strategy) ---
    if (len >= 4 && memcmp(buf, "BAD!", 4) == 0) {
        
        // --- CHECK 2: Specific Value (Needs Arithmetic/Interest strategy) ---
        // We look at the 5th byte. 
        if (len >= 5) {
            unsigned char val = buf[4];
            
            // Standard AFL might hit this eventually, but RL should prioritize arithmetic
            if (val == 0x42) { 
                
                // --- CHECK 3: Size Constraint (Needs "Block Delete" strategy) ---
                if (len == 10) {
                    crash();
                }
            }
        }
    }

    return 0;
}
