#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

// This function represents the "Constraint" we want to break
void check_input(char *buf, int len) {
    if (len < 5) return; // Level 1: Length Constraint

    // Level 2: Diversity Constraint (Needs wide value exploration)
    if (buf[0] == 'B') {
        // Level 3: Hard Constraint (Specific magic bytes)
        if (buf[1] == 'A' && buf[2] == 'D' && buf[3] == '!') {
            abort(); // CRASH! The Goal.
        }
    }
}

int main(int argc, char **argv) {
    char buf[100];
    ssize_t n = read(0, buf, 100);
    if (n > 0) {
        check_input(buf, n);
    }
    return 0;
}
