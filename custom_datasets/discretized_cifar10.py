import os
import pickle

import einops
import torch
import torchvision
from PIL import Image


class DummyVisionTokenizer:
  def __init__(self, vocab_size, image_size,
               add_mask_token=True,
               add_special_tokens=True):
    self.pad_token_id = None
    self.pad_token = None
    if add_mask_token:
      self.mask_token = vocab_size
      self.mask_token_id = vocab_size
      self.vocab_size = vocab_size + 1  # mask token
    else:
      self.vocab_size = vocab_size
    if add_special_tokens:
      self.bos_token_id = vocab_size
      self.bos_token = vocab_size
      self.eos_token_id = vocab_size + 1
      self.eos_token = vocab_size + 1
      self.vocab_size = self.vocab_size + 2  # mask token, bos_token, eos_token
    else:
      self.vocab_size = self.vocab_size
    self.image_size = image_size

  def __call__(self, x):
    return x

  def batch_decode(self, x):
    return einops.rearrange(x, "b (c h w) -> b c h w", c=3,
                     h=self.image_size)

  def decode(self, x):
    return einops.rearrange(x, "(c h w) -> c h w", c=3,
                     h=self.image_size)


class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
  def _check_integrity(self):
    batches_dir = os.path.join(self.root, self.base_folder)
    required_files = [
      'data_batch_1',
      'data_batch_2',
      'data_batch_3',
      'data_batch_4',
      'data_batch_5',
      'test_batch',
      'batches.meta',
    ]
    if os.path.exists(batches_dir):
      missing = [
        name for name in required_files
        if not os.path.exists(os.path.join(batches_dir, name))
      ]
      if missing:
        raise RuntimeError(
          'CIFAR-10 subset missing files: '
          + ', '.join(missing))
      return True
    return super()._check_integrity()

  def _load_meta(self):
    path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    if os.path.exists(path):
      with open(path, 'rb') as infile:
        data = pickle.load(infile, encoding='latin1')
        self.classes = data.get('label_names')
        if self.classes is None:
          raise RuntimeError('Dataset metadata missing label_names.')
        self.class_to_idx = {
          _class: i for i, _class in enumerate(self.classes)
        }
      return
    super()._load_meta()

  def __init__(self, root, train, **kwargs):
    super().__init__(root=root, train=train,
                     **kwargs)
    self.transform = torchvision.transforms.Compose(
      [
        torchvision.transforms.Resize(32),
        # torchvision.transforms.RandomHorizontalFlip(),  # Disabled mirroring
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
          lambda x: einops.rearrange(x, "c h w -> (c h w)")),
      ]
    )

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)
    img = (img * 255).to(torch.long)

    if self.target_transform is not None:
      target = self.target_transform(target)

    attention_mask = torch.ones_like(img)

    return {'input_ids': img, 'labels': target,
            'attention_mask': attention_mask}
