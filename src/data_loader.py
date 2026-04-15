from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_dataloaders(data_root, batch_size, aug_px=0):
    train_ops = []
    if aug_px > 0:
        train_ops.append(transforms.RandomCrop(32, padding=aug_px))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_transform = transforms.Compose(train_ops)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    testset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
