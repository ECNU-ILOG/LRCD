import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mlp(input_channel, output_channel, dtype, device):
    return nn.Sequential(
        nn.Linear(input_channel, 512, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        nn.Linear(512, 256, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        nn.Linear(256, output_channel, dtype=dtype).to(device)
    )

class extract_fea(nn.Module):
    def __init__(self, dataset, latent_dim, dtype, device):
        super(extract_fea, self).__init__()
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.emb_map = {}

        train_emb, test_emb = dataset.get_text_embedding(dtype, device)
        self.dict = {
            "train": train_emb,
            "test": test_emb
        }
        self.text_dim = self.dict["train"]["student"].shape[1]
        self.know = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.exer = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.stu = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.disc = get_mlp(self.text_dim, 1, dtype, device)
        self.emb = ...

        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_normal_(param)

    def extract(self, student_id, exercise_id, type):
        if type == "train":
            self.emb = self.dict["train"]
        else:
            self.emb = self.dict["test"]

        student_ts = self.stu(self.emb["student"])[student_id]
        disc_ts = self.disc(self.emb["disc"])[exercise_id]
        diff_ts = self.exer(self.emb["exercise"])[exercise_id]
        knowledge_ts = self.know(self.emb["knowledge"])

        return student_ts, diff_ts, disc_ts, knowledge_ts

    def update(self, item):
        self.eval()
        self.emb_map["student"] = self.stu(self.emb["student"])
        self.emb_map["knowledge"] = self.know(self.emb["knowledge"])
        return self.emb_map[item]
