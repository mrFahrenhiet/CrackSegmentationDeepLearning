from torchvision import transforms

train_loader_config = {
    'transforms': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ToTensor(),
    ]),
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 2
}

val_loader_config = {
    'transforms': transforms.Compose([transforms.Resize((320, 320)),
                                      transforms.ToTensor(),
                                      ]),
    'batch_size': 8,
    'shuffle': False,
    'num_workders': 2
}
