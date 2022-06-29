from torchvision import datasets, transforms
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(root="./data",
                                      train=True,
                                      transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]),
                                      download=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

class DataLoader:
    def __init__(self, dataset):
        super(data.DataLoader).__init__()
        self.dataloader = data.DataLoader(dataset=dataset,
                                     batch_size=64,
                                     shuffle=True,
                                     drop_last=True)