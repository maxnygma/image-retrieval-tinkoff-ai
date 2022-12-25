import cv2
from torch.utils.data import Dataset

from sklearn import model_selection


def split_data(data, current_fold):
    kfold = model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data['super_class_id'], groups=data['class_id'])):
        if fold == current_fold:
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            break

    return train_data, val_data


def preprocess_data(data):
    ''' Task-specific data preprocessing '''

    data['class_id'] = data['class_id'] - 1
    data['super_class_id'] = data['super_class_id'] - 1

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    return data


class RetrievalDataset(Dataset):
    def __init__(self, data, transforms=None, plot_embeddings=False):
        self.data = data
        self.transforms = transforms
        self.plot_embeddings = plot_embeddings

    def __getitem__(self, idx):
        img_path = 'data/stanford_image_retrieval/' + self.data['path'].iloc[idx]
        img = cv2.imread(img_path)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if not self.plot_embeddings:
            label = self.data['class_id'].iloc[idx]
        else:
            label = self.data['super_class_id'].iloc[idx]

        image_id = self.data['image_id'].iloc[idx]

        return {
            'data': img,
            'labels': label,
            'ids': image_id
        }

    def __len__(self):
        return len(self.data)