import torch
from torchvision import datasets, transforms


def data_generator(root, batch_size, train_p=0.8):
    # train_p is the percentage used to split the training dataset into training and validation dataset
    if 'CIFAR' in root:
        print('loading CIFAR10')
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root=root, train=True,
                                     download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False,
                                    download=True, transform=transform)

#        classes = ('plane', 'car', 'bird', 'cat',
#                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    else:
        print('loading MNIST')
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        
        train_dataset = datasets.MNIST(root=root, train=True, download=True,
                                   transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=True,
                                  transform=transform)

    train_size=int(len(train_dataset)*train_p)
    val_size=len(train_dataset)-train_size 
    print('{} train samples, {} validation samples, {} test samples'.format(train_size, val_size, len(test_set)))
    
    train_set, val_set=torch.utils.data.random_split(train_dataset,[train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, pin_memory=True)  
    print('{} train batches, {} validation batches, {} test batches'.format(len(train_loader),len(val_loader),len(test_loader)))
                  
    return train_loader, val_loader, test_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    root='data/MNIST'
    batch_size=64
    train_p=0.8
    train_loader, val_loader, test_loader=data_generator(root, batch_size, train_p)
    
    for data, target in train_loader:
        print(data.shape)
        break
    
    