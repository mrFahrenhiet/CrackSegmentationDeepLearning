import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import pandas as pd
import gc


class CrackData(Dataset):
    def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
        self.data = df
        self.img_transform = img_transforms
        self.mask_transform = mask_transform
        self.aux_transforms = aux_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
        mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

        if self.aux_transforms is not None:
            img = self.aux_transforms(img)

        seed = np.random.randint(420)

        random.seed(seed)
        torch.manual_seed(seed)

        img = transforms.functional.equalize(img)
        image = self.img_transform(img)

        random.seed(seed)
        torch.manual_seed(seed)

        mask = self.mask_transform(mask)

        return image, mask


class CrackDataTest(Dataset):
    def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
        self.data = df
        self.img_transform = img_transforms
        self.mask_transform = mask_transform
        self.aux_transforms = aux_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
        mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

        if self.aux_transforms is not None:
            img = self.aux_transforms(img)

        seed = np.random.randint(420)

        random.seed(seed)
        torch.manual_seed(seed)

        img = transforms.functional.equalize(img)
        image = self.img_transform(img)

        random.seed(seed)
        torch.manual_seed(seed)

        mask = self.mask_transform(mask)

        return image, mask, self.data['images'].iloc[idx]


if __name__ == '__main__':
    dfT = {
        'images': ['../temp/test_img.jpg'],
        'masks': ['../temp/test_mask.png']
    }

    dfT = pd.DataFrame.from_dict(dfT, )
    tfms = transforms.Compose([transforms.Resize((320, 320)),
                               transforms.ToTensor(),
                               ])

    dataset_train = CrackData(dfT, tfms, tfms, None)
    imge, msk = dataset_train[0]
    print(imge.shape, msk.shape)

    del imge
    del msk
    del dataset_train
    gc.collect()
