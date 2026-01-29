# [sysrel]

import json

def djb2_hash(data):
    """
    DJB2 Hash function.
    match the implementation in mutator.c exactly.
    """
    hash_val = 5381
    for byte in data:
        hash_val = ((hash_val << 5) + hash_val) + byte
        hash_val = hash_val & 0xFFFFFFFF
    return hash_val

# --- FEATURE VECTOR DEFINITION ---
# The RL Agent will see this vector: [CMP_TYPE, BIT_WIDTH, IS_CONST, DEPTH]
#
# 1. CMP_TYPE:  0.0=Unknown, 1.0=Integer Compare (CMP), 2.0=Memory Compare (memcmp/strcmp)
# 2. BIT_WIDTH: Normalized width (8-bit=0.1, 32-bit=0.4, 64-bit=0.8)
# 3. IS_CONST:  1.0 if comparing against a hardcoded constant, 0.0 if variable
# 4. DEPTH:     Normalized depth in the path (0.0 = start, 1.0 = deep)

constraints_db = {}

# --- CONSTRAINT 1: The "BAD!" Check (memcmp) ---
# Location: Start of main
# Code: if (memcmp(buf, "BAD!", 4) == 0) ...
#
# Static Analysis Attributes:
# - Type: Memory Compare (2.0)
# - Width: 32 bits / 4 bytes (0.4)
# - Const: Yes, "BAD!" is constant (1.0)
# - Depth: Start (0.0)
#
# Target Strategy: Dictionary or Havoc
feat_memcmp = [2.0, 0.4, 1.0, 0.0]

# We map likely starting inputs (Empty, 'A', Nulls) to this constraint
empty_hash = djb2_hash(b"") 
a_hash = djb2_hash(b"A")
trash_hash = djb2_hash(b"\x00" * 10)

constraints_db[str(empty_hash)] = feat_memcmp
constraints_db[str(a_hash)] = feat_memcmp
constraints_db[str(trash_hash)] = feat_memcmp


# --- CONSTRAINT 2: The 0x42 Check (val == 0x42) ---
# Location: Inside "BAD!" block
# Code: if (val == 0x42) ...
#
# Static Analysis Attributes:
# - Type: Integer Compare (1.0)
# - Width: 8 bits (0.1)
# - Const: Yes, 0x42 is constant (1.0)
# - Depth: Medium (0.5)
#
# Target Strategy: Arithmetic or Value Profile
feat_int8 = [1.0, 0.1, 1.0, 0.5]

bad_input = b"BAD!"
bad_hash = djb2_hash(bad_input)
constraints_db[str(bad_hash)] = feat_int8


# --- CONSTRAINT 3: The Length Check (len == 10) ---
# Location: Deepest nesting
# Code: if (len == 10) ...
#
# Static Analysis Attributes:
# - Type: Integer Compare (1.0)
# - Width: 32 bits (integer len) (0.4)
# - Const: Yes, 10 is constant (1.0)
# - Depth: Deep (1.0)
#
# Target Strategy: Block Deletion or Insertion (Size manipulation)
feat_len = [1.0, 0.4, 1.0, 1.0]

stage2_input = b"BAD!\x42"
stage2_hash = djb2_hash(stage2_input)
constraints_db[str(stage2_hash)] = feat_len


output_file = "constraints.json"
with open(output_file, "w") as f:
    json.dump(constraints_db, f, indent=4)

print(f"[+] Generated {output_file} with Realistic Static Analysis features.")
print(f"    - 'BAD!' Hash: {bad_hash} -> Features: {feat_int8}")
