import torch
import numpy as np
import random
from collections import defaultdict
from .options import args_parser

args = args_parser()


def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label = label.numpy()[0]
        else:
            label = label
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]

    return classes





"""
#edited so everybody has the same points
def sample_dirichlet_train_data(dataset, no_participants, no_samples=None, alpha=0.1):
    no_samples = None
    random.seed(args.manualseed)
    data_classes = build_classes_dict(dataset)
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    # Dirichlet-based allocation
    for n in range(no_classes):
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array([alpha] * no_participants))

        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]

    # Determine max number of samples each client can support
    if no_samples is None:
        no_samples = min(len(per_participant_list[i]) for i in range(no_participants))

    # Enforce exactly `no_samples` per client in per_participant_list
    for i in range(no_participants):
        client_data = per_participant_list[i]
        if len(client_data) >= no_samples:
            sampled_indices = np.random.choice(len(client_data), no_samples, replace=False)
            per_participant_list[i] = list(np.array(client_data)[sampled_indices])
        else:
            raise ValueError(f"Client {i} has only {len(client_data)} samples, but {no_samples} requested.")

    # Original logic for per_samples_list (subset of per_participant_list)
    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(len(per_participant_list[i]), no_samples, replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])

    return per_participant_list, per_samples_list




"""
#original
def sample_dirichlet_train_data(dataset, no_participants, no_samples, alpha=0.1):
    random.seed(args.manualseed)
    data_classes = build_classes_dict(dataset)
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    for n in range(no_classes):
        image_num = []
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))

        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))

            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]

    for i in range(len(per_participant_list)):
        no_samples = min(no_samples, len(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(len(per_participant_list[i]), no_samples,
                                        replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])

    return per_participant_list, per_samples_list

