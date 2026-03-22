# thuctapchuyennganh

# file data training
https://www.kaggle.com/datasets/vanthonguyen123/viet-ocr
# FUll code train
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ===== CHARSET =====
def build_charset():
    chars = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "áàảãạăắằẳẵặâấầẩẫậ"
        "đ"
        "éèẻẽẹêếềểễệ"
        "íìỉĩị"
        "óòỏõọôốồổỗộơớờởỡợ"
        "úùủũụưứừửữự"
        "ýỳỷỹỵ"
        " "
    )
    char2idx = {c: i+1 for i, c in enumerate(chars)}
    idx2char = {i+1: c for i, c in enumerate(chars)}
    return char2idx, idx2char


# ===== DATASET =====
class OCRDataset(Dataset):
    def __init__(self, img_dir, label_file, char2idx):
        self.samples = []
        self.char2idx = char2idx

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

                img_name, text = parts
                img_path = os.path.join(img_dir, img_name)

                if os.path.exists(img_path):
                    self.samples.append((img_path, text))

        print("Loaded:", len(self.samples))


    def resize_pad(self, img, target_h=32, target_w=512):
        h, w = img.shape
        new_w = int(w * (target_h / h))
        img = cv2.resize(img, (new_w, target_h))

        if new_w < target_w:
            pad = np.full((target_h, target_w - new_w), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        else:
            img = cv2.resize(img, (target_w, target_h))

        return img


    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        img = cv2.imread(img_path, 0)
        img = self.resize_pad(img)

        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        label = torch.tensor(self.encode(text), dtype=torch.long)

        return img, label, len(label)


# ===== COLLATE =====
def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)
    return imgs, labels, lengths


# ===== MODEL =====
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.rnn = nn.LSTM(128*8, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, -1)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x


# ===== PATH =====
DATA_PATH = "/content/data"


# ===== MAIN =====
device = torch.device("cuda")
print("Device:", device)

char2idx, idx2char = build_charset()
num_classes = len(char2idx) + 1

dataset = OCRDataset(
    os.path.join(DATA_PATH, "images"),
    os.path.join(DATA_PATH, "train_gt.txt"),
    char2idx
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

model = CRNN(num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CTCLoss(blank=0)


# ===== TRAIN =====
loss_history = []  # 🔥 thêm

for epoch in range(8):
    print(f"\nEpoch {epoch}")

    for i, (imgs, labels, label_lengths) in enumerate(loader):

        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        preds = preds.log_softmax(2)

        input_lengths = torch.full(
            size=(imgs.size(0),),
            fill_value=preds.size(1),
            dtype=torch.long
        )

        loss = criterion(
            preds.permute(1, 0, 2),
            labels,
            input_lengths,
            label_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item()) 
        if i % 50 == 0:
            print(f"Step {i} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"/content/model_epoch_{epoch}.pth")



np.save("/content/loss.npy", np.array(loss_history))

print("✅ Saved model + loss")
