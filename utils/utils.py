from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageFile

from torchvision import transforms

# This dataset includes extremely large source images; disable PIL's decompression
# bomb guard so they can be resized during preprocessing.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [f for f in self.files if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        last_error = None
        for offset in range(len(self.files)):
            current_idx = (idx + offset) % len(self.files)
            image_path = os.path.join(self.root, self.files[current_idx])
            try:
                with Image.open(image_path) as image:
                    image = image.convert('RGB')
                break
            except (OSError, ValueError) as error:
                last_error = error
        else:
            raise last_error
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    
    return feat_mean, feat_std