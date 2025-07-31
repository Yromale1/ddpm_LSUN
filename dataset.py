import os
import io
import lmdb
from torch.utils.data import Dataset
from PIL import Image

class MultiLMDBDataset(Dataset):
    def __init__(self, root_dir, split="train", max_per_class=None, transform=None):
        self.samples = []
        self.transform = transform
        
        # Parcours des dossiers LMDB par classe
        self.classes = sorted([d for d in os.listdir(root_dir) if d.endswith(f"{split}_lmdb")])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            lmdb_path = os.path.join(root_dir, class_name)
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                keys = [key for key, _ in cursor if not key.startswith(b'__')]
            
            # Limite le nombre d'échantillons par classe
            if max_per_class is not None:
                keys = keys[:max_per_class]
            
            # Enregistrer les paires (path_lmdb, key, class_idx)
            for key in keys:
                self.samples.append((lmdb_path, key, self.class_to_idx[class_name]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        lmdb_path, key, class_idx = self.samples[index]
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        with env.begin(write=False) as txn:
            img_bytes = txn.get(key)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_idx