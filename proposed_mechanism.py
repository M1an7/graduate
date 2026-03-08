import math

def RNS_ENCODE(x,r,moduli):

    #keep the first r digits
    x = int(x * (10**r)) / (10**r)
    x = math.floor(x* (10**r))
    encoded = []
    #encode to RNS + unary encoding
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

#find the first coprimes
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