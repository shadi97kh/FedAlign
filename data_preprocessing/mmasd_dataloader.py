# data_preprocessing/mmasd_data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
import os
import random

class MMASDDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = self._load_annotations(annotations_file)
        self.transform = transform

    def _load_annotations(self, annotations_file):
        data = []
        with open(annotations_file, 'r') as file:
            for line in file:
                cleaned_parts = clean_formatting(line)
                if cleaned_parts is None:
                    continue

                video_name, skeleton_name, action_label, asd_label = cleaned_parts
                data.append({
                    'video_path': video_name,
                    'label_path': skeleton_name,
                    'action_class': action_label,
                    'asd_class': asd_label
                })
        return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        video = self._load_video(item['video_path'])
        label = item['asd_class'] 

        if self.transform:
            video = self.transform(video)

        return video, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Adjust size as needed
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        frames = torch.from_numpy(frames).float() / 255.0  # Normalize
        return frames

def clean_formatting(line):
    """Cleans and formats each line, ensuring it has exactly 4 parts."""
    try:
        parts = line.strip().rsplit(' ', 3)
        if len(parts) != 4:
            print(f"Skipping line due to incorrect format: {line.strip()}")
            return None

        video_name, skeleton_name, action_label_str, asd_label_str = parts
        action_label = int(action_label_str)
        asd_label = int(asd_label_str)
        return video_name, skeleton_name, action_label, asd_label
    except ValueError as ve:
        print(f"Skipping line due to invalid label: {line.strip()} | Error: {ve}")
        return None

def distribute_balanced(label_data, num_clients):
    """
    Distributes data to clients ensuring each client gets at least one sample from each class.
    """
    clients = {i: [] for i in range(num_clients)}

    # First, distribute at least one sample from each class to each client
    for label, samples in label_data.items():
        random.shuffle(samples)
        for client_id in range(num_clients):
            if samples:
                clients[client_id].append(samples.pop())

    # Distribute remaining samples evenly among clients
    remaining_samples = []
    for samples in label_data.values():
        remaining_samples.extend(samples)

    random.shuffle(remaining_samples)
    for idx, sample in enumerate(remaining_samples):
        client_id = idx % num_clients
        clients[client_id].append(sample)

    return clients

def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size):
    annotations_file = os.path.join(data_dir, 'annotations.txt')

    # Read and clean the dataset
    label_name_dict = {}
    num_labels = 11
    for i in range(num_labels):
        label_name_dict[i] = []

    with open(annotations_file, 'r') as f:
        for line in f.readlines():
            cleaned_parts = clean_formatting(line)
            if cleaned_parts is None:
                continue

            video_name, skeleton_name, action_label, asd_label = cleaned_parts

            # Append data to the corresponding label
            label_name_dict[action_label].append({
                'video_path': video_name,
                'label_path': skeleton_name,
                'action_class': action_label,
                'asd_class': asd_label
            })

    # Distribute data among clients
    if partition_method == 'hetero':
        clients_data = distribute_balanced(label_name_dict, client_number)
    else:
        # Homogeneous partitioning
        data_list = []
        for samples in label_name_dict.values():
            data_list.extend(samples)
        random.shuffle(data_list)
        split_sizes = [len(data_list) // client_number] * client_number
        for i in range(len(data_list) % client_number):
            split_sizes[i] += 1
        clients_data = {}
        idx = 0
        for client_idx in range(client_number):
            clients_data[client_idx] = data_list[idx: idx + split_sizes[client_idx]]
            idx += split_sizes[client_idx]

    # Create datasets and dataloaders for each client
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx in range(client_number):
        client_data_list = clients_data[client_idx]
        client_dataset = MMASDDatasetFromList(client_data_list)
        data_local_num_dict[client_idx] = len(client_dataset)

        train_size = int(0.8 * len(client_dataset))
        test_size = len(client_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            client_dataset, [train_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_data_local_dict[client_idx] = train_loader
        test_data_local_dict[client_idx] = test_loader

    # Global data loaders (optional)
    # You can define train_data_global and test_data_global if needed
    train_data_global = None
    test_data_global = None

    # Number of classes
    class_num = 2  # For ASD classification

    return None, None, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num

class MMASDDatasetFromList(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video = self._load_video(item['video_path'])
        label = item['asd_class'] 

        if self.transform:
            video = self.transform(video)

        return video, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Adjust size as needed
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        frames = torch.from_numpy(frames).float() / 255.0  # Normalize
        return frames