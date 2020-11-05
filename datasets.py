class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.
        subsample : int
            Only load every |subsample| number of images.
        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0