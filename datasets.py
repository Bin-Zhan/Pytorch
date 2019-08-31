import os
import numpy as np
from torch.utils.data import Dataset



class Split2TnV():
    """
    Used to help split images into train set and validation set.

    Args:
        root (string): Path to the root folder.
        th (int): Threshold used to control class imbalance. Default: 0.
    """

    def __init__(self, root, th=0):

        self.root = root
        self.th = th
        self.counter = 0
        self.train = open('train.txt', 'w')
        self.val = open('val.txt', 'w')

    def closeFile(self):

        self.train.close()
        self.val.close()

    def creatList(self):

        for root, dirs, files in os.walk(self.root):
            if len(files) > self.th:
                valIdx = np.random.randint(0, len(files), min(int(np.ceil(len(files)*0.1)), 2))
                for i in range(len(files)):
                    if i in valIdx:
                        self.val.write(os.path.join(root, files[i]) + ' ' + str(self.counter) + '\n')
                    else:
                        self.train.write(os.path.join(root, files[i]) + ' ' + str(self.counter) + '\n')

                self.counter += 1
                if self.counter % 100 == 0:
                    print('proceed {} calsses'.format(self.counter))

        self.closeFile()



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):

    """
    Checks if a file is an allowed extension.
    Args:
        filename (string): Path to a file.
        extensions (tuple of strings): Extensions to consider (lowercase).
    Returns:
        bool: True if the filename ends with one of given extensions
    """

    return filename.lower().endswith(extensions)

def is_image_file(filename):

    """
    Checks if a file is an allowed image extension.
    Args:
        filename (string): Path to a file.
    Returns:
        bool: True if the filename ends with a known image extension.
    """

    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

class ItemList(Dataset):
    """
    Custom pytorch Dataset

    Args:
        txt_file (string): Path to txt file containing image path and label.
        transform: Torchvision transforms used for image preprocessing and augmentation. Default: None.

    Returns:
        tuple: Tensors of image and label.
    """

    def __init__(self, txt_file, transform=None):

        with open(txt_file, 'r') as f:
            self.lines = f.readlines()
        self.transform = transform

    def __len__(self):

        return len(self.lines)

    def __getitem__(self, idx):

        img_path = self.lines[idx].strip().split()[:-1]
        if isinstance(img_path, list):
            img_path = ' '.join(img_path)

        if is_image_file(img_path):
            try:
                img = Image.open(img_path)
                label = int(self.lines[idx].strip().split()[-1])
                label = torch.tensor(label)

                if self.transform:
                    img = self.transform(img)
            except OSError:
                print('xxxxxxxxxx Image file is corrupted: {} xxxxxxxxxx'.format(img_path))
                return self[(idx+1) % self.__len__()]
        else:
            return self[(idx+1) % self.__len__()]

        return img, label
