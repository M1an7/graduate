from math import gcd
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import sys

"""
python3 comm_cost.py r

Example usage:  
"""
def RLE_compress(j): #we just the numbers of ones
    return j.bit_length()

def pairwise_coprime_numbers(n):
    if n <= 0:
        return []
    
    coprime_list = [2]  # Start with 2, which is coprime
    num = 2
    
    while len(coprime_list) < n:
        if all(gcd(num, x) == 1 for x in coprime_list):
            coprime_list.append(num)
        num += 1
    return coprime_list


if (len(sys.argv)!=2):
    print("Usage: python3 comm_cost.py [r]")
    exit()

r = int(sys.argv[1])
FONTSIZE = 25
MODEL_PARAMETERS_CIFAR100 = 11237432


r_list = []
clients_list = []
num_rounds_list = []
spdz_cost = []
comm_cost_RNS = []
comm_cost_VANILLA=[]
comm_cost_RLE_compress = []

#### First plot: Communication cost in terms of r
for clients in range(5,54,5):
    clients_list.append(clients)
    comm_cost_VANILLA.append(32)#assuming standard 32 bit binary encoding
    max_number = 10**(r) -1
    n =0 
    while(True):
        n+=1
        M = 1
        first_coprimes = pairwise_coprime_numbers(n)
        for val in first_coprimes:
            M = M * val
        if max_number * clients <  int((M-1)/2):
            num_rounds = n
            break
    #print("Co-primes are", first_coprimes)

    comm_cost_RNS.append(sum(first_coprimes))
    cost_RLE_compress = 0
    for j in first_coprimes:
        cost_RLE_compress+= RLE_compress(j)
    comm_cost_RLE_compress.append(cost_RLE_compress)

    cost_per_share =  math.ceil (math.log2(clients*10**(r)))
    num_rounds_list.append(num_rounds) 
    spdz_cost.append(max(cost_per_share * (clients-1),32))


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.figure(0,figsize=(8, 6))

r_lit = r'$r$'
t_lit = r'$t=n-1$'
ss_label = "SA (" + t_lit+")"
plt.plot(clients_list,spdz_cost,marker='o', label=ss_label, color="orange")
plt.plot(clients_list,comm_cost_RNS,marker='o', label="Alg. 1 (Semi-Honest Shuffler)", color="darkgreen")
plt.plot(clients_list,comm_cost_RLE_compress,marker='o', label="Alg. 1 + Compress. (Honest Shuffler)", color="mediumspringgreen")
plt.plot(clients_list,comm_cost_VANILLA, marker='o', label="Vanilla FL", color = "orchid")
#plt.xticks(clients_list, fontsize = 15)
plt.xlabel("Number of Clients", fontsize = FONTSIZE)
plt.yticks(fontsize=15)
plt.ylabel("Communication Cost (bits)", fontsize = FONTSIZE)

#plt.xlabel(""+r_lit, fontsize = FONTSIZE)
plt.legend(fontsize = 14, markerscale=1)

#plt.ylim(0,200)
pdf_name = "r="+str(r)+".pdf"
plt.savefig(pdf_name, format="pdf", dpi=300, bbox_inches="tight")
plt.show()


#### Second plot: Expansion Factor

init_cifar_100_size = MODEL_PARAMETERS_CIFAR100 * 32 #in bits
cifar_100_expansion_factor = []
cifar_100_expansion_factor_RLE = []
cifar_100_accuracy = [15.87,48.15,51.46,52.6,52.8,53.6,54.21,55.45]
clients = 10
for r in range (8):
    r_list.append(r)
    num_rounds =0
    max_number = 10**(r) -1
    n =0 
    while(True):
        n+=1
        M = 1
        first_coprimes = pairwise_coprime_numbers(n)
        for val in first_coprimes:
            M = M * val
        if max_number * clients <  int((M-1)/2):
            num_rounds = n
            break
        cost = math.ceil(math.log2(clients* ( 2*10**(r) -1)))

    cost = sum(first_coprimes)
    cost_RLE_compress = 0
    for j in first_coprimes:
        cost_RLE_compress+= RLE_compress(j)

    cifar_100_expansion_factor.append((cost*MODEL_PARAMETERS_CIFAR100)/init_cifar_100_size)
    cifar_100_expansion_factor_RLE.append((cost_RLE_compress*MODEL_PARAMETERS_CIFAR100)/init_cifar_100_size)
    print(cifar_100_expansion_factor_RLE)

plt.plot(cifar_100_expansion_factor,cifar_100_accuracy,marker='o', label="Alg. 1 (Semi-Honest Shuffler)", color="darkgreen")
plt.plot(cifar_100_expansion_factor_RLE,cifar_100_accuracy,marker='o', label="Alg. 1 + Compress. (Honest Shuffler)", color="mediumspringgreen")



plt.axhline(y=55.45, color='crimson', linestyle='--', label ="Vanilla FL")
plt.legend(fontsize = 14, markerscale=1, loc = "lower right")
plt.xlabel("Expansion Factor", fontsize = FONTSIZE)
plt.ylabel("Corresponding Accuracy", fontsize = FONTSIZE)

plt.xticks(cifar_100_expansion_factor)
ticks = cifar_100_accuracy[:4] + cifar_100_accuracy[6:8]
plt.yticks(ticks)
file_name = "expansion_factor=" + str(r) +".pdf"
plt.savefig(file_name, format="pdf", bbox_inches="tight")
plt.show()

