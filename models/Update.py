
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from tqdm import tqdm

def gaussian_noise(data_shape, clip_constant, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * clip_constant, data_shape).to(device)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, shadow = False, PERCN_OF_SHADOW = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        db_splited = DatasetSplit(dataset, idxs)


        if shadow == True:
            db_to_use = [db_splited[i] for i in range(int(PERCN_OF_SHADOW * len(db_splited)))]
            self.ldr_train = DataLoader(db_to_use, batch_size=self.args.local_bs, shuffle=True)

        else:
            db_to_use = [db_splited[i] for i in range(int(PERCN_OF_SHADOW * len(db_splited)),len(db_splited))]
            self.ldr_train = DataLoader(db_to_use, batch_size=self.args.local_bs, shuffle=True)
           
        #print ("self ldr train is", len(self.ldr_train))

    def train(self, net):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep), desc="Local Training", leave=False):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.ldr_train, desc=f"Epoch {iter}", leave=False)):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                loss.backward()
                ##### DP #####
                """
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) #gradient clipping

                for name, param in net.named_parameters(): #allocation of current gradient to noised_gradient variable
                    clipped_grads[name] += param.grad
                for name, param in net.named_parameters(): #current gradient+gaussain_noise 
                    clipped_grads[name]+=gaussian_noise(clipped_grads[name].shape, 0.25, sigma, self.args.device)
                for name, param in net.named_parameters():
                    clipped_grads[name]/=self.args.local_bs
                for name, param in net.named_parameters(): #allocation of noised gradient to model gradient
                    temp=clipped_grads[name].squeeze(dim=0)
                    param.grad = temp
                """

                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

class LocalUpdateESIA(LocalUpdate):
    """
    带有 ESIA 主动攻击的本地训练类。
    训练完成后对模型参数进行主动注入（梯度上升），用于源推理攻击。
    """
    def __init__(self, args, dataset=None, idxs=None, shadow=False, PERCN_OF_SHADOW=None,
                 esia_target_x=None, esia_target_y=None, esia_Z_aug=None,
                 esia_eta=0.005, esia_eta_prime=0.005):
        super().__init__(args, dataset, idxs, shadow, PERCN_OF_SHADOW)
        self.esia_target_x = esia_target_x  # 目标样本 x
        self.esia_target_y = esia_target_y  # 目标样本 y
        self.esia_Z_aug = esia_Z_aug        # 增强样本列表
        self.esia_eta = esia_eta
        self.esia_eta_prime = esia_eta_prime

    def train(self, net):
        # 标准本地训练
        state_dict, loss = super().train(net)
        net.load_state_dict(state_dict)

        # ESIA主动注入（算法1第9行）
        if self.esia_target_x is not None and self.esia_target_y is not None:
            # 直接调用 Sia.py 里的 ESIA._inject_on_global
            from models.Sia import ESIA
            # 构造一个临时 ESIA 实例，仅用于注入
            esia_args = self.args
            esia_args.esia_eta = self.esia_eta
            esia_args.esia_eta_prime = self.esia_eta_prime
            esia = ESIA(args=esia_args)
            net = esia._inject_on_global(net, self.esia_target_x, self.esia_target_y, self.esia_Z_aug)
        return net.state_dict(), loss