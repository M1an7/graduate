# Supplementary Material for "Protection Against Source Inference Attacks in Federated Learning"


## Description
This folder contains the necessary scripts to run the experiments of the paper.

Note that the initial code for SIAs was taken from https://github.com/HongshengHu/SIAs-Beyond_MIAs_in_Federated_Learning (Source Inference Attacks: Beyond Membership Inference Attacks in Federated Learning). 
We expand their code to include our proposed solution and the reconstruction attacks of Section 5.


## Files

1.  Folders "data", "models", "utils"

	Implement the FL architecture and the SIA (authors: https://github.com/HongshengHu/SIAs-Beyond_MIAs_in_Federated_Learning with  odifications)
2. proposed_mechanism.py

   Implements Algorithm 1

4. main_recon.py

    Uses the reconstruction attacks of Section 5 and the proposed solution of Section 6 for SIAs.

5. comm_cost.py

    Compares the communication cost of the proposed mechanisms (Figure 2 and 10). 

6. computation_time.py

Measures the computation overhead of Algorithm 1. 

##### Datasets:
We use the MNIST, CIFAR-10 and CIFAR-100 dataset, which are automatically downloaded if they are not present in the directory. 


### Experiments: 
#### Proposed solution and reconstruction attacks against standard shuffling (Figures 1,3,6,7 and Tables 2,3 )

The experiment can be run using:
```bash
python3 main_recon.py --dataset=[DB] --model=cnn --alpha=[alpha] --local_ep=[local_epochs] --num_users=[number of clients] --epochs=[global epochs] --mode=[shuffling level to attack]
```
where:
- DB = MNIST/CIFAR10/CIFAR100
- alpha = The level of disimilarity 
- shuffling level to attack = MODEL/LAYER/PARAMETER (to respectively execute Algorithms 2,3 and 5).


Example for model-level shuffling on the MNIST dataset with alpha = 0.1, 10 local epochs, 10 global epochs and 10 clients:
```bash
python3  main_recon.py --dataset=MNIST --model=cnn --alpha=0.1 --local_ep=10 --num_users=10 --epochs=10 --mode="MODEL" 

```

#### Communication Cost (Figure 2 and 10)

The experiment can be run using:
```bash
python3 comm_cost.py [r]
```

#### Computation Cost (Table 1)

The experiment can be run using:
```bash
python3 computation_time.py  [NUM_CLIENTS] [model_parameters] [r]
```
For instance for CIFAR-100 with 10 clients:
```bash
python3 computation_time.py  10 11237432 4
```
