from torchvision import transforms

# Planes dataset
FGVC_PATH = "/home/x/xuxilie/data/fgvc-aircraft-2013b/"

# Oxford Flowers dataset
FLOWERS_PATH = "/tmp/datasets/oxford_flowers_pytorch/"

# DTD dataset
DTD_PATH="/home/x/xuxilie/data/dtd/"

# Stanford Cars dataset
CARS_PATH = "/home/x/xuxilie/data/cars_new"

# SUN397 dataset
SUN_PATH="/home/x/xuxilie/data/SUN397/splits_01/"

# FOOD dataset
FOOD_PATH = "/home/x/xuxilie/data/food-101"

# BIRDS dataset
BIRDS_PATH = "/home/x/xuxilie/data/birdsnap"

# PETS dataset
PETS_PATH = "/tmp/datasets/pets"

# Caltech datasets
CALTECH101_PATH = "/tmp/datasets"
CALTECH256_PATH = "/tmp/datasets"

# Data Augmentation defaults
TRAIN_TRANSFORMS = transforms.Compose([
            # transforms.Resize(32),
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        # transforms.CenterCrop(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
