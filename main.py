import argparse

import numpy as np
import pandas as pd
from huggingface_hub import login
from sklearn.metrics import balanced_accuracy_score
from plip import PLIP
from utils import set_random_seeds, get_visual_mbeddings, get_prompt_classifier

parser = argparse.ArgumentParser(description = "MI-VisionShot")
parser.add_argument('--folder', type=str)
parser.add_argument('--n_seeds', type=int, default=5)
parser.add_argument('--n_shots', type=int, default=16)
parser.add_argument('--k_patches', type=int, default=200)
parser.add_argument('--hug_token', type=str)
args = parser.parse_args()

login(args.hug_token) # Log in to HuggingFace

print("[INFO] Computing text embeddings")
# Texts prompts for TCGA RCC classes
classes = ["clear cell renal cell carcinoma",
         "chromophobe renal cell carcinoma",
         "papillary renal cell carcinoma"]
texts = ["A histopathology image of a " + cls for cls in classes]  # Image captions
plip = PLIP('vinid/plip')  # Load PLIP model
text_embeddings = plip.encode_text(texts, batch_size=1)  # Compute + normalize text embeddings
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

print("[INFO] Reading dataframe")
data = pd.read_csv("assets/TCGA_RCC.csv", delimiter=",")
list_WSI = data['WSI'].values
labels = data['GT'].values

print("[INFO] Loading visual embeddings")
bgap_embeddings, wght_embeddings = get_visual_mbeddings(args.folder, list_WSI, labels, args.k_patches, text_embeddings)
labels = np.stack(labels) # Ground truth labels
n_classes = len(np.unique(labels)) # Number of classes

print("[INFO] MI-VisionShot prediction framework")
list_acc = []
for seed in range(args.n_seeds): # Seed iteration in few_shot tunning
    set_random_seeds(seed_value=seed)

    # Few-shot random sample selection
    ids = np.array(list(range(len(labels))))
    train_ids = []
    for cls in range(n_classes):
        train_ids += list(np.random.choice(ids[labels == cls], size=args.n_shots, replace=False))
    test_ids = list(set(train_ids).symmetric_difference(ids))
    train_data, test_data = wght_embeddings[train_ids], bgap_embeddings[test_ids]
    train_labels, test_labels = labels[train_ids], labels[test_ids]

    # MI-SimpleShot classifier
    prompt_classifier = get_prompt_classifier(train_data, train_labels)
    test_pred = np.argmax(test_data @ prompt_classifier.T, axis=1)
    acc = round(balanced_accuracy_score(test_labels, test_pred), 4)  # Balanced ACC
    list_acc.append(acc)
acc = np.mean(list_acc)
print(f"[INFO] Balanced ACC: {acc:.4f}")