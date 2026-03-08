import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm


# we use prediction loss to conduct our attacks
# prediction loss: for a given sample (x, y), every local model will has a prediction loss on it. we consider the party who has the smallest prediction loss owns the sample.


def _safe_prob(probs, small_value=1e-30):
    return np.maximum(probs, small_value)

def uncertainty(probability, n_classes):
    uncert = []
    for i in range(len(probability)):
        unc = (-1 / np.log(n_classes)) * np.sum(probability[i] * np.log(_safe_prob(probability[i])))
        uncert.append(unc)
    return uncert


def entropy_modified(probability, target):
    entr_modi = []
    for i in range(len(probability)):
        ent_mod_1 = (-1) * (1 - probability[i][int(target[i])]) * np.log(_safe_prob(probability[i][int(target[i])]))
        probability_rest = np.delete(probability[i], int(target[i]))
        ent_mod_2 = -np.sum(probability_rest* np.log(_safe_prob(1 - probability_rest)))
        ent_mod = ent_mod_1 + ent_mod_2
        entr_modi.append(ent_mod)
    return entr_modi


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users

    def attack(self, net):
        correct_loss = 0
        len_set = 0
        for idx in tqdm(self.dict_mia_users, desc="SIA Attack", leave=False):

            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_mia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)

            y_loss_all = []

            # evaluate each party's training data on each party's model
            for local in self.dict_mia_users:

                y_losse = []

                idx_tensor = torch.tensor(idx)
                net.load_state_dict(self.w_locals[local])
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        data, target = data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.cuda()
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the test sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    y_losse.append(y_loss.cpu().detach().numpy())



                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            y_loss_all = torch.tensor(y_loss_all).to(self.args.gpu)

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)
            #print(f"Correct Loss for user with idx {idx} is {correct_local_loss}")

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        #print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
        #                                                                                          accuracy_loss))

        return accuracy_loss

    def attack_client(self, idx,net):
        correct_loss = 0
        len_set = 0
        dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_mia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)

        y_loss_all = []

        # evaluate each party's training data on each party's model
        for local in self.dict_mia_users:

            y_losse = []

            idx_tensor = torch.tensor(idx)
            net.load_state_dict(self.w_locals[local])
            net.eval()
            for id, (data, target) in enumerate(dataset_local):
                if self.args.gpu != -1:
                    data, target = data.cuda(), target.cuda()
                    idx_tensor = idx_tensor.cuda()
                log_prob = net(data)
                # prediction loss based attack: get the prediction loss of the test sample
                loss = nn.CrossEntropyLoss(reduction='none')
                y_loss = loss(log_prob, target)
                y_losse.append(y_loss.cpu().detach().numpy())



            y_losse = np.concatenate(y_losse).reshape(-1)
            y_loss_all.append(y_losse)

        y_loss_all = torch.tensor(y_loss_all).to(self.args.gpu)

        # test if the owner party has the largest prediction probability
        # get the parties' index of the largest probability of each sample
        index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
        correct_local_loss = index_of_party_loss.eq(
            idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

        correct_loss += correct_local_loss
        len_set += len(dataset_local.dataset)
        #print(f"Correct Loss for user with idx {idx} is {correct_local_loss}")

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        #print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
        #                                                                                          accuracy_loss))

        return accuracy_loss
    

import copy
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 你已有的
# from utils import DatasetSplit

class ESIA(object):
    """
    ESIA: Enhanced SIA in FedAvg (aligned with Algorithm 1 in the figure)

    Required external objects:
      - DatasetSplit(dataset, indices): returns a dataset subset
      - self.w_locals: dict {client_id: state_dict} for the target round (or each round)
      - self.dataset: full dataset
      - self.dict_mia_users: dict {client_id: list(indices)} membership sets / "source owner" sets
      - net: a torch.nn.Module with load_state_dict()

    Extra required for STRICT multi-round alignment (Algorithm 1 line 4..):
      - args.w_global_list: list of global model state_dict for rounds [0..T] or [1..T]
        If not provided, we run a single-round ESIA using the provided `net` as theta_t.

    Args expected fields (set defaults if missing):
      - local_bs (batch size)
      - gpu (-1 or cuda id)
      - esia_T (int, number of rounds to simulate; default len(w_global_list)-1 if provided else 1)
      - esia_eta (float, eta in line 9)
      - esia_eta_prime (float, eta' in line 9)
      - esia_n_aug (int, |Z_aug|)
      - esia_target_pick (str: "first" or "random")
      - esia_target_index (int)
      - esia_aug_mode (str: "mnist" / "cifar" / "none")
      - esia_seed (int)
    """

    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users

        # ---- ESIA hyperparams (Algorithm 1 line 9) ----
        self.eta = float(getattr(args, "esia_eta", 0.005))
        self.eta_prime = float(getattr(args, "esia_eta_prime", 0.005))
        self.n_aug = int(getattr(args, "esia_n_aug", 16))

        # multi-round control
        self.seed = int(getattr(args, "esia_seed", 0))
        self.target_pick = getattr(args, "esia_target_pick", "first")
        self.target_index = int(getattr(args, "esia_target_index", 0))

        self.aug_mode = getattr(args, "esia_aug_mode", "mnist")  # or "cifar", "none"

        # optional global states list for strict Algorithm 1 loop
        self.w_global_list = getattr(args, "w_global_list", None)
        if self.w_global_list is not None:
            # if list includes theta_0..theta_T, T = len-1
            self.T = int(getattr(args, "esia_T", max(1, len(self.w_global_list) - 1)))
        else:
            self.T = int(getattr(args, "esia_T", 1))  # fall back to 1-round ESIA

        self._rng = random.Random(self.seed)

    # ------------------------
    # Helper: pick a single target record z (Algorithm 1 line 3)
    # ------------------------
    def _pick_target_record(self):
        """
        Choose one (data, label) pair z from a target client.
        We pick from dict_mia_users keys (client ids).
        """
        client_ids = list(self.dict_mia_users.keys())
        if len(client_ids) == 0:
            raise ValueError("dict_mia_users is empty")

        if self.target_pick == "random":
            cid = self._rng.choice(client_ids)
            idx = self._rng.choice(self.dict_mia_users[cid])
        else:
            # deterministic: pick a client by index in sorted list
            client_ids_sorted = sorted(client_ids)
            cid = client_ids_sorted[self.target_index % len(client_ids_sorted)]
            idx_list = self.dict_mia_users[cid]
            idx = idx_list[0] if len(idx_list) > 0 else self._rng.choice(idx_list)

        x, y = self.dataset[idx]
        return cid, x, y

    # ------------------------
    # Helper: build augmented set Z_aug (Algorithm 1 line 3)
    # ------------------------
    def _augment(self, x):
        """
        Return one augmented version of x.
        Keep it simple and deterministic-ish; feel free to replace with your pipeline.
        """
        if self.aug_mode == "none":
            return x

        # x is a tensor [C,H,W] for vision datasets in typical FL codebases
        # We'll do small random affine/noise in a lightweight way without torchvision dependency.
        # (If you already use torchvision, replace with TF.affine etc.)
        xt = x.clone()

        if xt.dim() == 3 and xt.size(-1) >= 8:
            # random small translation by padding+crop
            C, H, W = xt.shape
            pad = 2
            xt_pad = torch.nn.functional.pad(xt.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
            dx = self._rng.randint(-2, 2)
            dy = self._rng.randint(-2, 2)
            x0 = pad + dx
            y0 = pad + dy
            xt = xt_pad[:, y0:y0 + H, x0:x0 + W]

        # small gaussian noise
        noise_std = 0.05 if self.aug_mode in ["mnist", "cifar"] else 0.02
        xt = torch.clamp(xt + noise_std * torch.randn_like(xt), 0.0, 1.0)
        return xt

    def _build_Z_aug(self, x):
        return [self._augment(x) for _ in range(self.n_aug)]

    # ------------------------
    # Helper: compute loss on a batch (cross-entropy)
    # ------------------------
    def _loss_on_batch(self, net, x, y):
        # x: [B,C,H,W]  y: [B]
        log_prob = net(x)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(log_prob, y)
        return loss  # shape [B]

    # ------------------------
    # Algorithm 1 line 9: gradient ascent injection on global model
    # ------------------------
    def _inject_on_global(self, net, z_x, z_y, Z_aug):
        """
        theta <- theta + eta * grad l(theta; z) + eta' * mean grad l(theta; z')
        Implemented in-place on `net` parameters.
        """
        device = torch.device("cpu") if self.args.gpu == -1 else torch.device(f"cuda:{self.args.gpu}")
        net.train()
        net.to(device)

        # prepare tensors
        x = z_x.unsqueeze(0).to(device)  # [1,C,H,W]
        y = torch.tensor([int(z_y)], dtype=torch.long, device=device)

        # grad for z
        net.zero_grad(set_to_none=True)
        loss_z = nn.CrossEntropyLoss()(net(x), y)
        loss_z.backward()
        grads_z = [p.grad.detach().clone() if p.grad is not None else None for p in net.parameters()]

        # grad for Z_aug mean
        net.zero_grad(set_to_none=True)
        if len(Z_aug) > 0:
            loss_sum = 0.0
            for xa in Z_aug:
                xa = xa.unsqueeze(0).to(device)
                loss_sum = loss_sum + nn.CrossEntropyLoss()(net(xa), y)
            loss_aug = loss_sum / float(len(Z_aug))
            loss_aug.backward()
            grads_a = [p.grad.detach().clone() if p.grad is not None else None for p in net.parameters()]
        else:
            grads_a = [None for _ in net.parameters()]

        # manual ascent
        with torch.no_grad():
            for p, gz, ga in zip(net.parameters(), grads_z, grads_a):
                if gz is not None:
                    p.add_(self.eta * gz)
                if ga is not None:
                    p.add_(self.eta_prime * ga)

        return net

    # ------------------------
    # Algorithm 1 line 10-13: compute \bar{l}_k^t and infer source
    # ------------------------
    def _infer_source(self, net, z_x, z_y, Z_aug):
        """
        For each client k, load theta_k (local model), compute:
          \bar{l}_k = l(theta_k; z) + mean_{z' in Z_aug} l(theta_k; z')
        Predict argmin_k \bar{l}_k
        Return predicted client id.
        """
        device = torch.device("cpu") if self.args.gpu == -1 else torch.device(f"cuda:{self.args.gpu}")

        # build a mini-batch containing z and aug (all labeled with z_y)
        X_list = [z_x] + Z_aug
        Y_list = [int(z_y)] * len(X_list)
        X = torch.stack(X_list, dim=0).to(device)  # [1+|Z_aug|, C,H,W]
        Y = torch.tensor(Y_list, dtype=torch.long, device=device)

        scores = []
        for local in self.dict_mia_users:
            net.load_state_dict(self.w_locals[local])
            net.eval()
            net.to(device)
            with torch.no_grad():
                losses = self._loss_on_batch(net, X, Y)  # [1+aug]
            lz = float(losses[0].item())
            la = float(losses[1:].mean().item()) if losses.numel() > 1 else 0.0
            scores.append((local, lz + la))

        pred = min(scores, key=lambda t: t[1])[0]
        return pred

    # ------------------------
    # Public API: attack(net) -> accuracy (ASR, %)
    # ------------------------
    def attack(self, net):
        """
        每个用户随机选取 100 个样本进行攻击，统计总正确率。
        """
        correct = 0
        total = 0
        for true_owner in self.dict_mia_users:
            user_indices = self.dict_mia_users[true_owner]
            if len(user_indices) == 0:
                continue
            # 随机选取 100 个样本（如不足则全选）
            if len(user_indices) > 100:
                sample_indices = self._rng.sample(user_indices, 100)
            else:
                sample_indices = user_indices
            for z_id in sample_indices:
                z_x, z_y = self.dataset[z_id]
                Z_aug = self._build_Z_aug(z_x)

                theta = copy.deepcopy(net)
                if self.w_global_list is not None and len(self.w_global_list) > 0:
                    theta.load_state_dict(self.w_global_list[0])

                for t in range(1, self.T + 1):
                    if self.w_global_list is not None and t < len(self.w_global_list):
                        theta.load_state_dict(self.w_global_list[t])
                    theta = self._inject_on_global(theta, z_x, z_y, Z_aug)
                    pred = self._infer_source(theta, z_x, z_y, Z_aug)
                    correct += int(pred == true_owner)
                    total += 1
        asr = 100.0 * correct / max(1, total)
        return asr

    # ------------------------
    # Public API: attack_client(idx, net) -> accuracy (ASR, %)
    # ------------------------
    def attack_client(self, idx, net):
        """
        Run ESIA for a specified true owner client idx (to mirror SIA.attack_client).
        We pick z from that client's membership set, generate Z_aug, then run T rounds.
        """
        if idx not in self.dict_mia_users or len(self.dict_mia_users[idx]) == 0:
            return 0.0

        # pick z from that client's set
        z_id = self.dict_mia_users[idx][0]
        z_x, z_y = self.dataset[z_id]
        Z_aug = self._build_Z_aug(z_x)

        theta = copy.deepcopy(net)
        if self.w_global_list is not None and len(self.w_global_list) > 0:
            theta.load_state_dict(self.w_global_list[0])

        correct = 0
        total = 0
        for t in range(1, self.T + 1):
            if self.w_global_list is not None and t < len(self.w_global_list):
                theta.load_state_dict(self.w_global_list[t])

            theta = self._inject_on_global(theta, z_x, z_y, Z_aug)
            pred = self._infer_source(theta, z_x, z_y, Z_aug)
            correct += int(pred == idx)
            total += 1

        return 100.0 * correct / max(1, total)
