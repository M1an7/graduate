import copy
import csv
import numpy as np
import torch
import random
import math
import struct
import time
import statistics
from models.Fed import FedAvg
from models.Nets import MLP, Mnistcnn, CifarCnn
from models.Sia import SIA,ESIA, DatasetSplit
from models.Update import LocalUpdate, LocalUpdateESIA
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
from models.test import test_fun,test_fun_topk
from sklearn.linear_model import LinearRegression
from proposed_mechanism import *

# Import ISR functionality
from ISR import compute_asr, square_patch_trigger

"""
Original file by Hongsheng Hu et al. 
from work Source Inference Attacks: Beyond Membership Inference Attacks in Federated Learning
We modified their code to include the reconstruction attacks [Section 4]. 
"""

#fixed parameters:
MAX_RUNS = 1 #number of executions
PERCN_OF_SHADOW = 0.05 #default: 0.05 percentage of the shadow dataset that the owner has
top_k = 1 #default:1 finds top_k accuracy, 
r = 4 #default:4 digits after the floating point to take into account


#####
#Debugging
#the following lines are used to execute only parts of the code, to speed up the process
RUN_SIA = 1 #default: 1 runs the source infernece attacks
RUN_RECONSTRUCTION = 1 #default: 1 runs the reconstruction attacks of Section 5
RUN_PROPOSED_MECHANISM = 1 #default: 1 runs algorithm 1 (proposed solution)
RUN_ACCURACY = 1 #default 1 finds the accuracy of the proposed mechanism
pern_of_parameter_to_reconc= 1 #default: 1 percentage of the parameters that will be reconstructed from the final layer, decrease this to speed up the execution
SMALLEST_EPOCH_TO_START_REMAPPING = -1 #default:-1 the epoch that we will start doing the SIAs attacks, for complex models like ResNet increase this to 3 to speed up the execution time

#####

names_of_last_fc = [] 
layers_to_remap =  []

#we keep in this dictionary the parameters which we have remapped
clients_remap_parameters = []


remapped_model = {} #we keep in this disctionary the models we have remapped
param_multplier = 1 #used when printing at the end



def reconstruct_model(w_locals, net_glob,dataset_train,  dict_sample_user, TARGET_CLIENT,correct_position, shadow_dataset):
    """
    Reconstructs the model by choosing 
    the one with the highest accuracy on the shadow dataset
    among the possible choices
    """
    best_accuracy = 0
    best_position = -1
    remapped_model[TARGET_CLIENT] = []
    test_model = copy.deepcopy(net_glob)
    # loop through all the models
    # and keep the one that has the highest accuracy
    # on the shadow dataset of TARGET_CLIENT

    for i in range (len(w_locals)):
        test_model.load_state_dict(w_locals[i])
        test_model.eval()
        accuracy = accuracy_on_target_data(test_model,shadow_dataset[TARGET_CLIENT])
        #print(f'Model {i} Accuracy: {accuracy}%')
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_position = i

    remapped_model[TARGET_CLIENT].append(best_position)
    return (best_position==correct_position)



def reconstruct_model_parameter(w_locals, net_glob,dataset_train,  dict_sample_user, TARGET_CLIENT,args,w_glob,shadow_dataset):
    """
    Reconstructs the model 
    by choosing the best accuracy
    of each parameter of the last layer, FedAvg the others (same as reconstruct_model)
    """

    corrects = 0
    totals = 0
    print("*** Remapping parameters of client ", TARGET_CLIENT)
    w_with_param = copy.deepcopy(w_glob)
    test_model = copy.deepcopy(net_glob)
    


    flattened_params = {
        i: {
            layer: torch.flatten(w_locals[i][layer]).tolist()
            for layer in layers_to_remap
        }
        for i in range(len(w_locals))
    }
    #find the possible choices for each parameter/layer

    for layer_name in layers_to_remap:
        remapped_parameters = []
        #print("currently in" , layer_name)
        #print("Parameters in ", layer_name, "are:", len(parameters))
        w_with_param = test_model.state_dict()

        times_avg = []
    

        #quick approximation
        """
        for param in range(len(flattened_params[0][layer_name])):
            #pick a random position
            randomness = random.random()
            if (randomness>0.3):
                random_pos = random.randint(0,len(w_locals)-1)
                remapped_parameters.append(flattened_params[random_pos][layer_name][param])
            #remap correct
            else:
                remapped_parameters.append(flattened_params[TARGET_CLIENT][layer_name][param])
        """
        #slower, actually do the remapping
        for param in range(int(len(flattened_params[0][layer_name]) * pern_of_parameter_to_reconc)):
            #print("-----")
            possible_choices = np.array([flattened_params[i][layer_name][param] for i in range(len(w_locals))])

            is_similar = np.allclose(possible_choices, np.mean(possible_choices), rtol=0.00001, atol=0.0001)

            if (is_similar):
                remapped_parameters.append(possible_choices[0])
                continue
            best_accuracy = 0
            best_position = -1
            
            for i in range(len(possible_choices)):
                flat_index = np.unravel_index(param, w_with_param[layer_name].shape)
                chosen_value = torch.tensor(possible_choices[i], dtype=w_with_param[layer_name].dtype, device=w_with_param[layer_name].device)
                
                w_with_param[layer_name][flat_index] = chosen_value
                test_model.load_state_dict(w_with_param)
                accuracy = accuracy_on_target_data(test_model, shadow_dataset[TARGET_CLIENT])
                #print("i is", i, "accuracy is", accuracy, "on value", chosen_value)
                # Check for the best accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_position = i

            # choose the best parameter value
            chosen = possible_choices[best_position]
            remapped_parameters.append(float(chosen))
            flat_index = np.unravel_index(param, w_with_param[layer_name].shape)
            chosen_value = torch.as_tensor(chosen, dtype=w_with_param[layer_name].dtype, device=w_with_param[layer_name].device)
            w_with_param[layer_name][flat_index] = chosen_value


            # Check if the chosen parameter matches the target client or is close enough
            #if (best_position == TARGET_CLIENT) or abs(chosen - flattened_params[TARGET_CLIENT][layer_name][param]) <= 0.0001:
            #    corrects += 1
        # Store the remapped parameters for the target client
        clients_remap_parameters[TARGET_CLIENT][layer_name] = remapped_parameters
        test_model.load_state_dict(w_with_param)
        test_model.eval()

    return corrects



def accuracy_on_target_data(test_model, shadow_dataset):
    """
    Computes accuracy of test_model on the shadow dataset.
    Optimized for performance.
    """
    correct = 0
    total = 0

    test_model.eval()
    with torch.no_grad():
        for images, labels in shadow_dataset:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = test_model(images)

            _, topk_preds = outputs.topk(top_k, dim=1, largest=True, sorted=True)
            correct += torch.sum(torch.any(topk_preds == labels.view(-1, 1), dim=1)).item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = 100 * correct / total if total != 0 else 0
    return accuracy


def avg(alist):
    return sum(alist)/len(alist)


# parse args
args = args_parser()
print("users = ", args.num_users)
print("alpha =", args.alpha)
print("Local Epochs=", args.local_ep)
MODE = args.mode # MODEL: shuffles per model | LAYER: shuffle per layer | PARAMETER: shuffler per param

if (MODE not in ["MODEL", "LAYER", "PARAMETER"]):
    print("Reconstruction attack should run by setting mode to either MODEL, LAYER or PARAMETER")
    exit()

print(">", MODE, " Reconstruction Attack!")



args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

num_of_succeses = 0
SIA_attacks_before =[]
SIA_attacks_after = []
SIA_attacks_shadow = []
SIA_attacks_prop_defense = []
# ISR lists will store true injection success rates (percentage)
ISR_before = []  # ISR for vanilla (no defense)
ISR_prop_defense = []  # ISR for algorithm 1 defense

executed_rounds = MAX_RUNS

total_distances = []
total_size_weights = []
total_SIAs = []



for run in range(MAX_RUNS):
    clients_remap_parameters = []
    for i in range(args.num_users): #change this to number of users:
        remaped_parameters_tuple = {}

        for layer in layers_to_remap:
            remaped_parameters_tuple[layer] = []
        clients_remap_parameters.append(remaped_parameters_tuple)

    remapped_model = {}
    print("[Run: ", run, "/", MAX_RUNS, "]")
    # load dataset and split data for users
    dataset_train, dataset_test, dict_party_user, dict_sample_user = get_dataset(args)

    # build model

    if args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
        names_of_last_fc = ["fc3.weight", "fc3.bias"] 
        layers_to_remap =  ["fc3.weight","fc3.bias"]

    elif args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CifarCnn(args=args).to(args.device)
        names_of_last_fc = ["fc3.weight", "fc3.bias"] 
        layers_to_remap =  ["fc3.weight","fc3.bias"]

    elif args.model == 'cnn' and args.dataset == 'CIFAR100': 
        #net_glob = CifarCnn(args=args).to(args.device) 
        net_glob = models.resnet18(pretrained=False) 
        num_features = net_glob.fc.in_features 
        net_glob.fc = nn.Linear(num_features, 100) 
        net_glob = net_glob.to(args.device)

        names_of_last_fc = ["fc.weight", "fc.bias"] 
        layers_to_remap =  ["fc.weight", "fc.bias"] 

    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        img_size = dataset_train[0][0].shape

        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        names_of_last_fc =  ["layer_hidden.weight", "layer_hidden.bias"]
        layers_to_remap =["layer_hidden.weight", "layer_hidden.bias"]
    else:
        exit('Error: unrecognized model')

    empty_net = net_glob
    net_glob.train()
    net_glob_encoded = copy.deepcopy(net_glob)
    total_parameters = 0

    #w_glob = {name: torch.zeros_like(param) for name, param in net_glob.named_parameters()}
    size_per_client = []
    for i in range(args.num_users):
        size = len(dict_party_user[i])
        size_per_client.append(size)
        #print("Size of client ",i, "is ", size)
    total_size = sum(size_per_client)
    size_weight = np.array(np.array(size_per_client) / total_size)
    # copy weights
    w_glob =  net_glob.state_dict()
    encoded_w_glob = net_glob_encoded.state_dict()
    ### training
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        w_locals_esia = [w_glob for i in range(args.num_users)]  # 初始化 ESIA 模型列表
        w_locals_encoded = [w_glob for i in range(args.num_users)]
    

    skip=0
    best_SIA_vanilla =0 
    best_SIA_of_recon =0 

    for curr_epoch in range(args.epochs):
        print("***** EPOCH:",curr_epoch,"******")
        #local_updates = []
        #loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_locals_esia = []  # 用于 ESIA 的注入模型
        for idx in range(args.num_users):
            try:
                print("--- Training client ", idx)
                # 构造 ESIA 所需目标样本和增强样本
                user_indices = dict_party_user[idx]
                if len(user_indices) > 0:
                    z_id = user_indices[0]
                    z_x, z_y = dataset_train[z_id]
                    from models.Sia import ESIA
                    esia_n_aug = getattr(args, "esia_n_aug", 16)
                    esia = ESIA(args=args)
                    Z_aug = esia._build_Z_aug(z_x)
                else:
                    z_x = z_y = Z_aug = None
                
                # 为 SIA 进行标准训练（不注入）
                local_sia = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx], shadow=False, PERCN_OF_SHADOW=PERCN_OF_SHADOW)
                w_sia, loss_sia = local_sia.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w_sia)
                else:
                    w_locals.append(copy.deepcopy(w_sia))
                
                # 为 ESIA 进行带注入的训练
                local_esia = LocalUpdateESIA(args=args, dataset=dataset_train, idxs=dict_party_user[idx], shadow=False, PERCN_OF_SHADOW=PERCN_OF_SHADOW,
                                            esia_target_x=z_x, esia_target_y=z_y, esia_Z_aug=Z_aug)
                w_esia, loss_esia = local_esia.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals_esia[idx] = copy.deepcopy(w_esia)
                else:
                    w_locals_esia.append(copy.deepcopy(w_esia))

                #run for our mechanism
                if (RUN_ACCURACY==1):
                    local = LocalUpdateESIA(args=args, dataset=dataset_train, idxs=dict_party_user[idx], shadow=False, PERCN_OF_SHADOW=PERCN_OF_SHADOW,
                                           esia_target_x=z_x, esia_target_y=z_y, esia_Z_aug=Z_aug)
                    w, loss = local.train(net=copy.deepcopy(net_glob_encoded).to(args.device))
                    if args.all_clients:
                        w_locals_encoded[idx] = copy.deepcopy(w)
                    else:
                        w_locals_encoded.append(copy.deepcopy(w))
            except Exception as e:
                print("Error occured when training the model, probably the dirichlet distribution returned no values for a client")
                print(e)
                skip = 1
                break
        if skip == 1:
             print("Skipping this round")
             executed_rounds -=1
             continue


        w_glob = FedAvg(w_locals, size_weight)
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        # shuffle models
        order = [i for i in range(len(w_locals))]
        random.shuffle(order)

        new_w_locals = []

        if (curr_epoch <=SMALLEST_EPOCH_TO_START_REMAPPING):
            continue   

        # 记录每轮实验结果到 output/main_recon_results.csv
        # SIA 模型评估
        from models.Sia import SIA, ESIA
        w_glob_sia = FedAvg(w_locals, size_weight) if not args.all_clients else w_locals[0]
        net_glob_sia = copy.deepcopy(net_glob)
        net_glob_sia.load_state_dict(w_glob_sia)
        net_glob_sia.eval()
        sia_train_acc, _ = test_fun_topk(net_glob_sia, dataset_train, args, top_k=top_k)
        sia_test_acc, _ = test_fun_topk(net_glob_sia, dataset_test, args, top_k=top_k)
        
        # ESIA 模型评估
        w_glob_esia = FedAvg(w_locals_esia, size_weight) if not args.all_clients else w_locals_esia[0]
        net_glob_esia = copy.deepcopy(net_glob)
        net_glob_esia.load_state_dict(w_glob_esia)
        net_glob_esia.eval()
        esia_train_acc, _ = test_fun_topk(net_glob_esia, dataset_train, args, top_k=top_k)
        esia_test_acc, _ = test_fun_topk(net_glob_esia, dataset_test, args, top_k=top_k)
        
        # SIA/ESIA 攻击评估
        w_locals_map = {i: w_locals[i] for i in range(len(w_locals))}
        w_locals_esia_map = {i: w_locals_esia[i] for i in range(len(w_locals_esia))}
        dict_mia_subset = {i: dict_sample_user[i] for i in range(len(dict_sample_user))}
        sia_net = copy.deepcopy(net_glob)
        sia = SIA(args=args, w_locals=w_locals_map, dataset=dataset_train, dict_mia_users=dict_mia_subset)
        sia_asr = sia.attack(sia_net)
        esia_net = copy.deepcopy(net_glob)
        esia = ESIA(args=args, w_locals=w_locals_esia_map, dataset=dataset_train, dict_mia_users=dict_mia_subset)
        esia_asr = esia.attack(esia_net)
        
        # 打印 SIA/ESIA 成功率和模型准确率
        print("==================================================")
        print("[Vanilla FL - Standard Training (SIA)]")
        print(f"  SIA Attack Success Rate (ASR): {sia_asr:.2f}%")
        print(f"  Training Accuracy: {sia_train_acc:.2f}%")
        print(f"  Testing Accuracy: {sia_test_acc:.2f}%")
        print("[Vanilla FL - ESIA Training (with Injection)]")
        print(f"  ESIA Attack Success Rate (ASR): {esia_asr:.2f}%")
        print(f"  Training Accuracy: {esia_train_acc:.2f}%")
        print(f"  Testing Accuracy: {esia_test_acc:.2f}%")
        print("==================================================")
        
        # 初始化记录，用于综合CSV输出，所有字段默认为N/A
        epoch_records = {
            'epoch': curr_epoch,
            'sia_vanilla_asr': 'N/A',
            'sia_vanilla_train': 'N/A',
            'sia_vanilla_test': 'N/A',
            'esia_vanilla_asr': 'N/A',
            'esia_vanilla_train': 'N/A',
            'esia_vanilla_test': 'N/A',
            'sia_recon_asr': 'N/A',
            'sia_recon_train': 'N/A',
            'sia_recon_test': 'N/A',
            'esia_recon_asr': 'N/A',
            'esia_recon_train': 'N/A',
            'esia_recon_test': 'N/A',
            'sia_alg1_asr': 'N/A',
            'sia_alg1_train': 'N/A',
            'sia_alg1_test': 'N/A',
            'esia_alg1_asr': 'N/A',
            'esia_alg1_train': 'N/A',
            'esia_alg1_test': 'N/A',
            'isr_vanilla': 'N/A',
            'isr_alg1': 'N/A',
        }
        
        # 填充无防御Vanilla FL的数据
        sia_asr_val = sia_asr.item() if hasattr(sia_asr, 'item') else float(sia_asr)
        esia_asr_val = esia_asr.item() if hasattr(esia_asr, 'item') else float(esia_asr)
        epoch_records['sia_vanilla_asr'] = round(sia_asr_val, 2)
        epoch_records['sia_vanilla_train'] = round(sia_train_acc, 2)
        epoch_records['sia_vanilla_test'] = round(sia_test_acc, 2)
        epoch_records['esia_vanilla_asr'] = round(esia_asr_val, 2)
        epoch_records['esia_vanilla_train'] = round(esia_train_acc, 2)
        epoch_records['esia_vanilla_test'] = round(esia_test_acc, 2)
        # compute true ISR (injection success) on ESIA-trained vanilla model using a simple patch trigger
        target_label = 0
        # wrap dataset in DataLoader to ensure batches and tensor labels
        isr_val = compute_asr(net_glob_esia, torch.utils.data.DataLoader(dataset_test, batch_size=args.bs), square_patch_trigger, target_label, device=args.device) * 100
        ISR_before.append(isr_val)
        epoch_records['isr_vanilla'] = round(isr_val, 2)

        ## Defeat shuffling by mapping the models back to their original owners (Section 5)
        if RUN_RECONSTRUCTION == 1:
            if MODE == "MODEL":
                #shuffle all models
                for i in order:
                    new_w_locals.append(copy.deepcopy(w_locals[i]))
            elif MODE == "LAYER":
                #shuffle all the FC3's 
                #we will use the FedAvg for FC1/FC2/Conv
                for i in order:
                    a_model = copy.deepcopy(w_glob)
                    a_model[names_of_last_fc[0]] = copy.deepcopy(w_locals[i][names_of_last_fc[0]])
                    a_model[names_of_last_fc[1]] = copy.deepcopy(w_locals[i][names_of_last_fc[1]])
                    new_w_locals.append(a_model)
            elif MODE == "PARAMETER":
                new_w_locals = copy.deepcopy(w_locals)
            else:
                print("Wrong mode...")
                exit()

            #get the shadow_datasets
            shadow_dataset = []
            for client in range(len(w_locals)):
                db_splited = DatasetSplit(dataset_train, dict_party_user[client])
                db_to_use = [db_splited[i] for i in range(int(PERCN_OF_SHADOW * len(db_splited))+1)]
                shadow_dataset.append(DataLoader(db_to_use, batch_size=64, shuffle=False))

            if MODE in ["MODEL", "LAYER"]:
                for TARGET_CLIENT in range(len(w_locals)):
                    correct_position = order.index(TARGET_CLIENT)
                    success = reconstruct_model(new_w_locals, copy.deepcopy(net_glob), dataset_train, dict_party_user, TARGET_CLIENT,correct_position, shadow_dataset)
                    if success == True:
                        num_of_succeses+=1
            else:
                for TARGET_CLIENT in range(len(w_locals)): 
                    result = reconstruct_model_parameter(w_locals, copy.deepcopy(net_glob),dataset_train,  dict_party_user, TARGET_CLIENT,args,w_glob,shadow_dataset)
                    #print("Reconstructed", result, "of the model.")
                    num_of_succeses+=result

        #Run the SIA attack
        if (RUN_SIA == 1): 
            empty_net_3 = copy.deepcopy(empty_net)
            empty_net_4 = copy.deepcopy(empty_net)
            
            # 使用前面已计算的sia_asr值，避免重复评估
            SIA_result = max(round(sia_asr, 2) if not hasattr(sia_asr, 'item') else round(sia_asr.item(), 2), (1/args.num_users)*100)
            if (SIA_result>best_SIA_vanilla):
                best_SIA_vanilla = SIA_result
            print("[Vanilla FL] SIA accuracy :",SIA_result,"%")

            reconstructed_w_locals = []
            if RUN_RECONSTRUCTION == 0:
                reconstructed_w_locals = w_locals
            else:
                if (MODE == "MODEL"):
                    for i in range(len(w_locals)):
                        pos = remapped_model[i][0]
                        #print("Remapping user", i, "to ", pos)
                        reconstructed_w_locals.append(copy.deepcopy(new_w_locals[pos]))
                else:
                    reconstructed_w_locals = [copy.deepcopy(w_glob) for i in range(len(w_locals))]
                    if (MODE == "LAYER"):
                        reconstructed_w_locals = [copy.deepcopy(w_glob) for i in range(len(w_locals))]
                        #avg all layers but not FC3
                        for i in range(len(w_locals)):
                            pos = remapped_model[i][0]
                            reconstructed_w_locals[i][names_of_last_fc[0]] = copy.deepcopy(new_w_locals[pos][names_of_last_fc[0]])
                            reconstructed_w_locals[i][names_of_last_fc[1]] = copy.deepcopy(new_w_locals[pos][names_of_last_fc[1]])              
                    elif (MODE == "PARAMETER"):
                        #take the avg of every layer but
                        #take the selected values from the shadow model, stored in the dictionaries
                        #now reconstruct the correct parameters for some of the FC3 paramters
                        param_multplier=0 
                        for i in range(len(w_locals)):
                            for layer_name in layers_to_remap:
                                #find parameters
                                parameters = reconstructed_w_locals[0][layer_name].view(-1).tolist()
                                #param_multplier+= len(parameters)
                          
                                for param in range(int(len(parameters)*pern_of_parameter_to_reconc)):
                                    reconstructed_w_locals[i][layer_name].view(-1)[param] = clients_remap_parameters[i][layer_name][param]
                        
            #run the attack on the reconstructed
            
            SIA_attack = ESIA(args=args, w_locals=reconstructed_w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user) 
            attack_acc_after = SIA_attack.attack(net=empty_net_4.to(args.device))
            try:
                acc_val = attack_acc_after.item()
            except Exception:
                acc_val = float(attack_acc_after)
            SIA_result = max(round(acc_val,2),(1/args.num_users)*100)
            if (SIA_result>best_SIA_of_recon):
                best_SIA_of_recon = SIA_result
            print("[Reconstructed Models (Section 5)] SIA accuracy :",SIA_result,"%")
            
            # 计算重建模型的準確率
            test_model_recon = copy.deepcopy(net_glob)
            if MODE == "MODEL" and len(w_locals) > 0:
                test_model_recon.load_state_dict(reconstructed_w_locals[0])
            elif len(reconstructed_w_locals) > 0:
                test_model_recon.load_state_dict(reconstructed_w_locals[0])
            test_model_recon.eval()
            acc_train_recon, _ = test_fun_topk(test_model_recon, dataset_train, args, top_k=top_k)
            acc_test_recon, _ = test_fun_topk(test_model_recon, dataset_test, args, top_k=top_k)
            
            # ESIA 重建模型评估
            esia_attack_recon = ESIA(args=args, w_locals=reconstructed_w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user)
            esia_acc_recon = esia_attack_recon.attack(net=empty_net_3.to(args.device))
            try:
                esia_acc_recon_val = esia_acc_recon.item()
            except Exception:
                esia_acc_recon_val = float(esia_acc_recon)
            esia_acc_recon_val = max(round(esia_acc_recon_val, 2), (1/args.num_users)*100)
            
            # 记录重建后的数据
            epoch_records['sia_recon_asr'] = SIA_result
            epoch_records['sia_recon_train'] = round(acc_train_recon, 2)
            epoch_records['sia_recon_test'] = round(acc_test_recon, 2)
            epoch_records['esia_recon_asr'] = esia_acc_recon_val
            epoch_records['esia_recon_train'] = round(acc_train_recon, 2)
            epoch_records['esia_recon_test'] = round(acc_test_recon, 2)
            
            print("[Reconstructed Models (Section 5)] Training accuracy: {:.2f}%".format(acc_train_recon))
            print("[Reconstructed Models (Section 5)] Testing accuracy: {:.2f}%".format(acc_test_recon))
            print("[Reconstructed Models (Section 5)] ESIA accuracy: {:.2f}%".format(esia_acc_recon_val))
           
            

        if (RUN_PROPOSED_MECHANISM==1):
            co_primes = find_coprimes(args.num_users*((10**r)-1))
            joint_model = []

            #instead of encoding than summing as per Algorithm 8 it is faster to first sum and then encode, i.e. take the w_glob
            encoded_w_glob = copy.deepcopy(FedAvg(w_locals_encoded, size_weight))
            for layer_name in w_glob:
                parameters = torch.flatten(w_glob[layer_name]).tolist()
                for param in range(int(len(parameters))):
                    encoded_w_glob[layer_name].view(-1)[param] = RNS_DECODE(RNS_ENCODE(parameters[param],r,co_primes),r ,co_primes)
            
            # 评估 Alg.1 防御下的 SIA 攻击
            sia_attack_alg1 = SIA(args=args, w_locals=[encoded_w_glob for i in range(len(w_locals))], dataset=dataset_train, dict_mia_users=dict_sample_user) 
            sia_acc_alg1 = sia_attack_alg1.attack(net=copy.deepcopy(empty_net).to(args.device))
            try:
                sia_acc_alg1_val = sia_acc_alg1.item()
            except Exception:
                sia_acc_alg1_val = float(sia_acc_alg1)
            sia_acc_alg1_val = max(round(sia_acc_alg1_val, 2), (1/args.num_users)*100)
            
            # 评估 Alg.1 防御下的 ESIA 攻击
            esia_attack_alg1 = ESIA(args=args, w_locals=[encoded_w_glob for i in range(len(w_locals))], dataset=dataset_train, dict_mia_users=dict_sample_user) 
            esia_acc_alg1 = esia_attack_alg1.attack(net=copy.deepcopy(empty_net).to(args.device))
            try:
                esia_acc_alg1_val = esia_acc_alg1.item()
            except Exception:
                esia_acc_alg1_val = float(esia_acc_alg1)
            esia_acc_alg1_val = max(round(esia_acc_alg1_val, 2), (1/args.num_users)*100)
            
            net_glob_encoded.load_state_dict(encoded_w_glob)
            net_glob_encoded.eval()
            
            # 评估 Alg.1 防御下的模型准确率
            acc_train_alg1, _ = test_fun_topk(net_glob_encoded, dataset_train, args, top_k=top_k)
            acc_test_alg1, _ = test_fun_topk(net_glob_encoded, dataset_test, args, top_k=top_k)
            
            # 打印 Alg.1 防御下的 SIA/ESIA 成功率和准确率
            print("==================================================")
            print("[Algorithm 1 Defense - Protected Models]")
            print(f"  SIA Attack Success Rate (ASR): {sia_acc_alg1_val:.2f}%")
            print(f"  ESIA Attack Success Rate (ASR): {esia_acc_alg1_val:.2f}%")
            print(f"  Training Accuracy: {acc_train_alg1:.2f}%")
            print(f"  Testing Accuracy: {acc_test_alg1:.2f}%")
            print("==================================================")
            
            # 记录Alg.1防御下的数据
            epoch_records['sia_alg1_asr'] = sia_acc_alg1_val
            epoch_records['sia_alg1_train'] = round(acc_train_alg1, 2)
            epoch_records['sia_alg1_test'] = round(acc_test_alg1, 2)
            epoch_records['esia_alg1_asr'] = esia_acc_alg1_val
            epoch_records['esia_alg1_train'] = round(acc_train_alg1, 2)
            epoch_records['esia_alg1_test'] = round(acc_test_alg1, 2)
            
            # record ISR for Alg.1 defense using the same patch-trigger evaluation
            isr_alg1_val = compute_asr(net_glob_encoded, torch.utils.data.DataLoader(dataset_test, batch_size=args.bs), square_patch_trigger, target_label, device=args.device) * 100
            ISR_prop_defense.append(isr_alg1_val)
            epoch_records['isr_alg1'] = round(isr_alg1_val, 2)



        acc_train, loss_train_ = test_fun_topk(net_glob, dataset_train, args,top_k=top_k)
        acc_test, loss_test = test_fun_topk(net_glob, dataset_test, args,top_k=top_k)
        print("[Vanilla FL] Training accuracy of the joint model: {:.2f}%".format(acc_train))
        print("[Vanilla FL] Testing accuracy of the joint model: {:.2f}%".format(acc_test))
        print("------")
        
        # 写入综合CSV表格
        import os
        csv_path = 'output/attack_results_summary.csv'
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 写入表头（仅在文件不存在时）
            if not file_exists:
                writer.writerow([
                    'Epoch',
                    'SIA_Vanilla_ASR', 'SIA_Vanilla_Train', 'SIA_Vanilla_Test',
                    'SIA_Recon_ASR', 'SIA_Recon_Train', 'SIA_Recon_Test',
                    'SIA_Alg1_ASR', 'SIA_Alg1_Train', 'SIA_Alg1_Test',
                    'ESIA_Vanilla_ASR', 'ESIA_Vanilla_Train', 'ESIA_Vanilla_Test',
                    'ESIA_Recon_ASR', 'ESIA_Recon_Train', 'ESIA_Recon_Test',
                    'ESIA_Alg1_ASR', 'ESIA_Alg1_Train', 'ESIA_Alg1_Test',
                    'ISR_Vanilla', 'ISR_Alg1'
                ])
            
            # 写入数据行，缺失的字段用 N/A 填充
            writer.writerow([
                epoch_records.get('epoch', 'N/A'),
                epoch_records.get('sia_vanilla_asr', 'N/A'),
                epoch_records.get('sia_vanilla_train', 'N/A'),
                epoch_records.get('sia_vanilla_test', 'N/A'),
                epoch_records.get('sia_recon_asr', 'N/A'),
                epoch_records.get('sia_recon_train', 'N/A'),
                epoch_records.get('sia_recon_test', 'N/A'),
                epoch_records.get('sia_alg1_asr', 'N/A'),
                epoch_records.get('sia_alg1_train', 'N/A'),
                epoch_records.get('sia_alg1_test', 'N/A'),
                epoch_records.get('esia_vanilla_asr', 'N/A'),
                epoch_records.get('esia_vanilla_train', 'N/A'),
                epoch_records.get('esia_vanilla_test', 'N/A'),
                epoch_records.get('esia_recon_asr', 'N/A'),
                epoch_records.get('esia_recon_train', 'N/A'),
                epoch_records.get('esia_recon_test', 'N/A'),
                epoch_records.get('esia_alg1_asr', 'N/A'),
                epoch_records.get('esia_alg1_train', 'N/A'),
                epoch_records.get('esia_alg1_test', 'N/A'),
                epoch_records.get('isr_vanilla', 'N/A'),  # ISR_Vanilla
                epoch_records.get('isr_alg1', 'N/A')      # ISR_Alg1
            ])
    #append the maximum accuracy
    SIA_attacks_before.append(max(best_SIA_vanilla,(1/args.num_users)*100))
    SIA_attacks_after.append(max(best_SIA_of_recon,(1/args.num_users)*100))
    # Record ISR (ESIA attack success rate)
    ISR_before.append(max(esia_asr_val, (1/args.num_users)*100))

# experiment setting
exp_details(args)
print("MODE = ", MODE)
print("MAX RUNS (actually executed) = ", executed_rounds)


if RUN_SIA == 1:
    print("Initial SIA accuracy:", avg(SIA_attacks_before))
    print("After reconstruction SIA accuracy:", avg(SIA_attacks_after))
    print("Average ISR (vanilla, patch-trigger):", avg(ISR_before))

if RUN_PROPOSED_MECHANISM==1:
    print("SIA accuracy on proposed defense:", avg(SIA_attacks_prop_defense))
    print("Average ISR (Alg1 defense, patch-trigger):", avg(ISR_prop_defense))

print("SHADOW MODL PERCN: ", PERCN_OF_SHADOW )

print("\nwith db= ",args.dataset)
print("users = ", args.num_users)
print("alpha =", args.alpha)
print("Local Epochs=", args.local_ep)

# 打印输出文件位置
print("\n" + "="*60)
print("实验统计结果已保存到:")
print("  - 详细结果表格: output/attack_results_summary.csv")
print("="*60)

