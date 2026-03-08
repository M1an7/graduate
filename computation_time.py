from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import hashlib
import random
import copy
from Crypto.Util.Padding import pad, unpad
import time
import math
import sys



#usage:
#python3 computation_time.py  [NUM_CLIENTS] [model_parameters] [r]

#e.g. for cifar100
#python3 computation_time.py  10 11237432 4

if (len(sys.argv)!=4):
    print("Usage: python3 computation_time.py  [NUM_CLIENTS] [model_parameters] [r]")
    exit()


NUM_CLIENTS = int(sys.argv[1]) 
MODEL_PARAMETERS= int(sys.argv[2]) #size of the model
r = int(sys.argv[3]) #precision


def RNS_ENCODE(x,r,moduli):
    #x = math.floor(x* (10**r))
    x = int(x * (10**r)) / (10**r)
    x = math.floor(x* (10**r))
    encoded = []

    for m in moduli:
        residue = x%m
        unary_residue = residue * [1] + [0] * (m-residue)
        encoded.append(unary_residue)
    return encoded


# Function for RNS decoding using the Chinese Remainder Theorem (CRT)
def RNS_DECODE(unary_remainders,r,moduli):

    remainders = [sum(x) for x in unary_remainders]
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for i in range(len(moduli)):
        mi = moduli[i]
        ai = remainders[i]
        Mi = M // mi
        inv = mod_inverse(Mi, mi)
        result += ai * Mi * inv

    result = result % M  # Ensure the result is within the bounds of the product of moduli

    # Adjust to handle negative values
    if result > M // 2:  # We want to wrap the result into the correct range
        result -= M
    return result/10**(r)

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y


def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"No modular inverse for a={a} modulo m={m}")
    return x % m
def find_coprimes(x):
    coprimes = []
    product = 1
    num = 2 
    while product < x:
        if all(math.gcd(num, cp) == 1 for cp in coprimes):
            coprimes.append(num)
            product *= num
            if product >= x:
                break
        num += 1
    
    return coprimes


co_primes = find_coprimes(NUM_CLIENTS*((10**r)-1))
rounds = len(co_primes)

###measure encoding time on randomly generated parameters)
print("**** Measuring encoding time:")
elapsed_time_encoding = 0
start_time = time.time()
for param in range(MODEL_PARAMETERS):
    RNS_ENCODE(random.random(),r,co_primes) 
end_time = time.time()
elapsed_time_encoding = end_time-start_time
print("*Encoding: Elapsed time (seconds):", elapsed_time_encoding)

### measure decoding time on randomly generated parameters)
print("**** Measuring decoding time:")

elapsed_time_decoding = 0
for param in range(MODEL_PARAMETERS):
    a_random_encoded = RNS_ENCODE(random.random(),r,co_primes) 
    start_time = time.time()
    RNS_DECODE(a_random_encoded,r,co_primes)
    end_time = time.time()
    elapsed_time_decoding+= end_time-start_time
print("* Encoding: Elapsed time (seconds):", elapsed_time_decoding)
