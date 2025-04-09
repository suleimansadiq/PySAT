import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose4

# Load model parts
fc2_out = np.load("fc2_out.npy").astype(np.uint8)
fc3_W = np.load("fc3_W.npy").astype(np.uint8)
fc3_b = np.load("fc3_b.npy").astype(np.uint8)
add_lut = np.load("posit8_add.npy").astype(np.uint8)
mul_lut = np.load("posit8_mul.npy").astype(np.uint8)

print("[*] Loaded all input files.")

pool = IDPool()
cnf = CNF()

def bits8(name):
    return [pool.id(f"{name}_{i}") for i in range(8)]

# Add symbolic constant to symbolic value
def encode_add_const(sym_bits, const_val, name):
    out_bits = bits8(name)
    for a in range(256):
        a_bin = [(a >> i) & 1 for i in range(8)]
        cond = [sym_bits[i] if a_bin[i] else -sym_bits[i] for i in range(8)]
        result = add_lut[a, const_val]
        for i in range(8):
            bit_val = (result >> i) & 1
            cnf.append(cond + ([out_bits[i]] if bit_val else [-out_bits[i]]))
    return out_bits

# Add a list of 8-bit symbolic vectors
def encode_add_pairwise_chain(vectors, base_name):
    acc = vectors[0]
    for idx, vec in enumerate(vectors[1:], start=1):
        out = bits8(f"{base_name}_sum_{idx}")
        for a in range(256):
            for b in range(256):
                a_bin = [(a >> i) & 1 for i in range(8)]
                b_bin = [(b >> i) & 1 for i in range(8)]
                clause = []
                for i in range(8):
                    clause.append(acc[i] if a_bin[i] else -acc[i])
                    clause.append(vec[i] if b_bin[i] else -vec[i])
                result = add_lut[a, b]
                for i in range(8):
                    bit_val = (result >> i) & 1
                    cnf.append(clause + ([out[i]] if bit_val else [-out[i]]))
        acc = out
    return acc

# Enforce logit1 < constant
def encode_lt(sym_bits, const_val):
    for i in reversed(range(8)):
        bit_val = (const_val >> i) & 1
        if bit_val == 0:
            cnf.append([-sym_bits[i]])
        elif bit_val == 1:
            cnf.append([-sym_bits[i]])
            break

# Symbolic weights for class 1
N = 5  # How many symbolic weights to modify
W_sym = {}
mul_outs = []

for i in range(N):
    W_sym[i] = bits8(f"W_{i}")
    mul_out = bits8(f"mul_{i}")
    mul_outs.append(mul_out)
    for b in range(256):
        b_bin = [(b >> j) & 1 for j in range(8)]
        cond = [W_sym[i][j] if b_bin[j] else -W_sym[i][j] for j in range(8)]
        result = mul_lut[fc2_out[i], b]
        for k in range(8):
            bit_val = (result >> k) & 1
            cnf.append(cond + ([mul_out[k]] if bit_val else [-mul_out[k]]))

# Build symbolic logit[1]
logit1_sym = encode_add_pairwise_chain(mul_outs, "logit1")

# Add constant part of logit[1] (fixed weights from i=N to 83)
const_sum = fc3_b[1]
for i in range(N, 84):
    const_sum = add_lut[const_sum, mul_lut[fc2_out[i], fc3_W[i, 1]]]

logit1_sym = encode_add_const(logit1_sym, const_sum, "logit1_final")

# Compute constant logit[8]
logit8 = fc3_b[8]
for i in range(84):
    logit8 = add_lut[logit8, mul_lut[fc2_out[i], fc3_W[i, 8]]]

# Enforce: logit[1] < logit[8]
encode_lt(logit1_sym, logit8)

# Force at least one bit of any W_sym to change
diff_clause = []
for i in range(N):
    original_val = int(fc3_W[i, 1])
    orig_bits = [(original_val >> j) & 1 for j in range(8)]
    for j in range(8):
        diff_clause.append(-W_sym[i][j] if orig_bits[j] else W_sym[i][j])
cnf.append(diff_clause)

print("[*] CNF built. Solving...")

with Glucose4(bootstrap_with=cnf) as solver:
    if solver.solve():
        print("[+] SAT: Adversarial weight flip found (1 -> 8)")
        model = solver.get_model()
        for i in range(N):
            bits = [1 if v in model else 0 for v in W_sym[i]]
            new_val = sum([(b << k) for k, b in enumerate(bits)])
            old_val = int(fc3_W[i, 1])
            print(f"W[{i},1]: {old_val} -> {new_val}")
    else:
        print("[-] UNSAT: No flip found in given symbolic space.")
