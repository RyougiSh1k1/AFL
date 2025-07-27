import os
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.model_selection import train_test_split
import numpy as np
import ujson

def prepare_data(args):
    if args.dataset == "tinyimagenet":
        trainset, testset = tinyimagenet_dataset(args)
    elif args.dataset == "cifar100":
        trainset, testset = cifar100_dataset(args)
    elif args.dataset == "cifar10":
        trainset, testset = cifar10_dataset(args)
    else:
        trainset, testset = None, None
        print("Unavailable dataset!")
        return
    
    np.random.seed(args.seed)
    
    if args.dataset == "cifar100" and args.continual:
        # Special handling for CIFAR-100 continual learning setup
        data_idx_train, task_info, statistic = separate_data_continual(
            trainset, testset, args.num_clients, args.num_classes, 
            args.tasks_per_client, args.classes_per_task, args.seed)
        return trainset, data_idx_train, testset, task_info
    else:
        # Original code for other datasets
        data_idx_train, y, statistic = separate_data(
            trainset, testset, args.num_clients, args.num_classes,
            args.niid, args.balance, args.partition, args.alpha, args.shred)
        return trainset, data_idx_train, testset, None

def tinyimagenet_dataset(args):
    # Data loading code
    dir = os.path.join(args.data,'tiny-imagenet-200')
    traindir = os.path.join(dir, 'train')
    valdir = os.path.join(dir, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset

def cifar100_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])

    train_dataset = datasets.CIFAR100(
        root=args.data + "/cifar100", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(
        root=args.data + "/cifar100", train=False, download=True, transform=train_transform)

    return train_dataset, val_dataset

def cifar10_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])

    train_dataset = datasets.CIFAR10(
        root=args.data + "/cifar10", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=args.data + "/cifar10", train=False, download=True, transform=train_transform)

    return train_dataset, val_dataset

def separate_data_continual(data, data_test, num_clients, num_classes, tasks_per_client, classes_per_task, seed):
    """
    Create continual learning setup for CIFAR-100:
    - 10 clients
    - Each client has 4 tasks
    - Each task has 20 randomly selected classes
    """
    np.random.seed(seed)
    
    dataset_label_train = data.targets
    dataset_label = np.array(dataset_label_train)
    
    # Get indices for each class
    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[dataset_label == i])
    
    # Initialize storage
    dataidx_map = {}  # client_id -> list of indices
    task_info = {}    # client_id -> list of tasks, each task is a list of classes
    statistic = {}    # client_id -> statistics
    
    # All available classes
    all_classes = list(range(num_classes))
    
    for client in range(num_clients):
        client_indices = []
        client_tasks = []
        client_stats = []
        
        # Create tasks for this client
        for task in range(tasks_per_client):
            # Randomly select classes for this task
            task_classes = np.random.choice(all_classes, size=classes_per_task, replace=False)
            task_classes = sorted(task_classes.tolist())
            client_tasks.append(task_classes)
            
            # Collect all indices for these classes
            task_indices = []
            task_stats = []
            
            for class_id in task_classes:
                class_indices = idx_for_each_class[class_id]
                # Randomly sample some data from this class for this client
                # Each client gets approximately 1/num_clients of each class data
                num_samples = len(class_indices) // num_clients
                if num_samples > 0:
                    selected_indices = np.random.choice(class_indices, size=num_samples, replace=False)
                    task_indices.extend(selected_indices.tolist())
                    task_stats.append((int(class_id), len(selected_indices)))
            
            client_indices.extend(task_indices)
            client_stats.append(task_stats)
        
        dataidx_map[client] = np.array(client_indices)
        task_info[client] = client_tasks
        statistic[client] = client_stats
    
    # Convert to format expected by the rest of the code
    data_idx = []
    for client in range(num_clients):
        data_idx.append(torch.from_numpy(dataidx_map[client]))
    
    # Print statistics
    print("\n" + "="*50)
    print("Continual Learning Setup Summary:")
    print(f"Number of clients: {num_clients}")
    print(f"Tasks per client: {tasks_per_client}")
    print(f"Classes per task: {classes_per_task}")
    print("="*50)
    
    for client in range(num_clients):
        print(f"\nClient {client}:")
        for task_id, (task_classes, task_stats) in enumerate(zip(task_info[client], statistic[client])):
            print(f"  Task {task_id}: Classes {task_classes}")
            print(f"    Total samples: {sum([stat[1] for stat in task_stats])}")
    
    return data_idx, task_info, statistic

def separate_data(data, data_test, num_clients, num_classes, niid=False, balance=False, partition=None, alpha = 0.1,least_samples=1,class_per_client = 10):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_label_train = data.targets
    dataset_label = np.array(dataset_label_train)

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    data_idx = []
    for client in range(num_clients):
        idxs = dataidx_map[client]
        idxs = np.array(idxs)
        y[client] = dataset_label[idxs]
        data_idx.append(torch.from_numpy(idxs))
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data, data_test

    return data_idx, y, statistic

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)