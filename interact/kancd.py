import torch
import torch.nn as nn
import torch.nn.functional as F

class PosLinear(nn.Linear):  # 确保权重非负
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight  # relu=max(0,x)
        return F.linear(input, weight, self.bias)

class KANCD_IF(nn.Module):
    def __init__(self, know_n, latent_dim, dropout, device, dtype):
        super(KANCD_IF, self).__init__()
        self.knowledge_num = know_n
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.k_diff_full = PosLinear(self.latent_dim, 1, dtype=dtype).to(self.device)
        self.stat_full = PosLinear(self.latent_dim, 1, dtype=dtype).to(self.device)

        self.score_mlp = nn.Sequential(
            PosLinear(self.knowledge_num, 512, dtype=dtype).to(device),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(512, 256, dtype=dtype).to(device),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(256, 1, dtype=dtype).to(device),
            nn.Sigmoid()
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, student_ts, diff_ts, disc_ts, knowledge_ts, q_mask):
        batch, dim = student_ts.size()
        stu_emb = student_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        exer_emb = diff_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)

        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
                                            - torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)) * q_mask
        return self.score_mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        self.eval()
        blocks = torch.split(torch.arange(mastery.shape[0]).to(device=self.device), 5)
        mas = []
        for block in blocks:
            batch, dim = mastery[block].size()
            stu_emb = mastery[block].view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
            knowledge_emb = knowledge.repeat(batch, 1).view(batch, self.knowledge_num, -1)
            mas.append(torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1))
        return torch.vstack(mas)
