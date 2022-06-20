from torchvision import transforms

trainLoaderConfig = {
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

valLoaderConfig = {
    'transforms': transforms.Compose([transforms.Resize((320, 320)),
                                      transforms.ToTensor(),
                                      ]),
    'batch_size': 8,
    'shuffle': False,
    'num_workders': 2
}

modelConfig = {
    'encoderBackbone': 'efficientnet-b2'
}
