import os
import random

import numpy as np
import torch
from tqdm import tqdm

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_visual_mbeddings(folder, list_WSI, labels, k, text_embeddings):
    bgap_embeddings, wght_embeddings = [], []
    for it, file_name in enumerate(tqdm(list_WSI)):
        file_name = os.path.join(folder, file_name + ".npy")
        patch_embeddings = np.load(file_name)

        # BGAP embeddings
        wsi_embedding = np.mean(patch_embeddings, axis=0)
        wsi_embedding = wsi_embedding / np.linalg.norm(wsi_embedding, ord=2, axis=-1, keepdims=True)
        bgap_embeddings.append(wsi_embedding)

        # MI-VisionShot embeddings
        patch_sim = patch_embeddings @ text_embeddings[labels[it]]
        k = min(k, len(patch_sim))
        idx_sort = np.argsort(patch_sim)[::-1][:k]  # (1) Top-K most informative patches
        patch_embeddings = np.mean(patch_embeddings[idx_sort], axis=0)
        patch_embeddings = patch_embeddings / np.linalg.norm(patch_embeddings, ord=2, axis=-1, keepdims=True)

        wght_embeddings.append(patch_embeddings)
    bgap_embeddings = np.stack(bgap_embeddings)  # BGAP embeddings
    wght_embeddings = np.stack(wght_embeddings)  # MI-VisionShot embeddings
    return bgap_embeddings, wght_embeddings

def get_prompt_classifier(train_data, train_labels):
    prompt_classifier = []
    for class_idx in np.unique(train_labels):
        class_wsi = train_data[train_labels == class_idx]
        prompt_class = np.mean(class_wsi, axis=0)  # Prototype
        prompt_class = prompt_class / np.linalg.norm(prompt_class, ord=2, axis=-1, keepdims=True)
        prompt_classifier.append(prompt_class)
    prompt_classifier = np.stack(prompt_classifier)
    return prompt_classifier
