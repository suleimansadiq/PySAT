import numpy as np
import itertools
from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose4
from softposit import posit8

# Load data
fc2_out = np.load("fc2_out.npy").astype(np.uint8)
fc3_W = np.load("fc3_W.npy").astype(np.uint8)
fc3_b = np.load("fc3_b.npy").astype(np.uint8)
add_lut = np.load("posit8_add.npy")
mul_lut = np.load("posit8_mul.npy")
posit8_decode = np.load("posit8_float_decode.npy")

def bits8(name, pool):
    return [pool.id(f"{name}_{i}") for i in range(8)]

def encode_add_const(sym_bits, const_val, name, cnf, pool):
    out_bits = bits8(name, pool)
    for a in range(256):
        a_bin = [(a >> i) & 1 for i in range(8)]
        cond = [sym_bits[i] if a_bin[i] else -sym_bits[i] for i in range(8)]
        result = add_lut[a, const_val]
        for i in range(8):
            bit_val = (result >> i) & 1
            cnf.append(cond + ([out_bits[i]] if bit_val else [-out_bits[i]]))
    return out_bits

def encode_lt(sym_bits, const_val, cnf):
    for i in reversed(range(8)):
        bit_val = (const_val >> i) & 1
        if bit_val == 0:
            cnf.append([-sym_bits[i]])
        elif bit_val == 1:
            cnf.append([-sym_bits[i]])
            break

# Target weight: W[0,1]
i = 0
j = 1
orig_val = fc3_W[i, j]
orig_bits = [(orig_val >> k) & 1 for k in range(8)]
orig_bin = format(orig_val, '08b')
print("Testing W[0,1]")
print("Original value:", orig_val)
print("Float value   :", posit8_decode[orig_val])
print("Bit pattern   :", orig_bin)

attempt = 1

# Try flipping 1 to 8 bits
for k in range(1, 9):
    for bit_indices in itertools.combinations(range(8), k):
        print()
        print("Attempt", attempt)
        attempt += 1
        print("Flipping bits:", bit_indices)

        # Show expected flipped pattern
        trial_bits = orig_bits[:]
        for b in bit_indices:
            trial_bits[b] = 1 - trial_bits[b]
        trial_bin = ''.join(str(bit) for bit in reversed(trial_bits))
        print("Flipped pattern (expected):", trial_bin)

        pool = IDPool()
        cnf = CNF()
        W_bits = bits8("W", pool)

        # Fix all non-flipped bits
        for b in range(8):
            if b not in bit_indices:
                cnf.append([W_bits[b]] if orig_bits[b] else [-W_bits[b]])

        # Symbolic multiplication
        mul_out = bits8("mul", pool)
        for b in range(256):
            b_bin = [(b >> t) & 1 for t in range(8)]
            cond = [W_bits[t] if b_bin[t] else -W_bits[t] for t in range(8)]
            result = mul_lut[fc2_out[i], b]
            for t in range(8):
                bit_val = (result >> t) & 1
                cnf.append(cond + ([mul_out[t]] if bit_val else [-mul_out[t]]))

        # Precompute constant part of logit[1]
        const_sum = fc3_b[j]
        for idx in range(1, 84):
            const_sum = add_lut[const_sum, mul_lut[fc2_out[idx], fc3_W[idx, j]]]

        # Symbolic addition
        logit1 = encode_add_const(mul_out, const_sum, "logit1_final", cnf, pool)

        # Precompute logit[8]
        logit8 = fc3_b[8]
        for idx in range(84):
            logit8 = add_lut[logit8, mul_lut[fc2_out[idx], fc3_W[idx, 8]]]

        encode_lt(logit1, logit8, cnf)

        with Glucose4(bootstrap_with=cnf) as solver:
            if solver.solve():
                model = solver.get_model()
                new_val = sum([(1 if W_bits[b] in model else 0) << b for b in range(8)])
                float_old = posit8_decode[orig_val]
                float_new = posit8_decode[new_val]
                delta = float_new - float_old
                new_bin = format(new_val, '08b')

                print("New value:", new_val)
                print("Float old -> new:", float_old, "->", float_new)
                print("Binary old -> new:", orig_bin, "->", new_bin)
                print("Float delta:", delta)

                if new_val != orig_val:
                    print("SAT: misclassification occurred with value change")
                    exit(0)
                else:
                    print("SAT: satisfied, but weight did not change")
            else:
                print("UNSAT: flip did not cause misclassification")
