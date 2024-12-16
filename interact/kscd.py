import torch
import torch.nn as nn
import torch.nn.functional as F

class PosLinear(nn.Linear):  # 确保权重非负
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight  # relu=max(0,x)
        return F.linear(input, weight, self.bias)

class KSCD_IF(nn.Module):
    def __init__(self, know_n, latent_dim, dropout, device, dtype):
        super(KSCD_IF, self).__init__()
        self.knowledge_num = know_n
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.prednet_full1 = PosLinear(self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False,
                                       dtype=dtype).to(self.device)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False,
                                       dtype=dtype).to(self.device)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(1 * self.knowledge_num, 1, dtype=dtype).to(self.device)
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, student_ts, diff_ts, disc_ts, knowledge_ts, q_mask):
        stu_ability = torch.mm(student_ts, knowledge_ts.T).sigmoid()
        exer_diff = torch.mm(diff_ts, knowledge_ts.T).sigmoid()
        batch_stu_vector = stu_ability.repeat(1, self.knowledge_num).reshape(stu_ability.shape[0], self.knowledge_num,
                                                                             stu_ability.shape[1])
        batch_exer_vector = exer_diff.repeat(1, self.knowledge_num).reshape(exer_diff.shape[0], self.knowledge_num,
                                                                            exer_diff.shape[1])

        kn_vector = knowledge_ts.repeat(stu_ability.shape[0], 1).reshape(stu_ability.shape[0], self.knowledge_num,
                                                                         self.latent_dim)

        preference = torch.tanh(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.tanh(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * q_mask.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(q_mask, dim=1).unsqueeze(1)
        y_pd = sum_out / count_of_concept
        return y_pd.view(-1)

    def transform(self, mastery, knowledge):
        stu_mastery = torch.mm(mastery, knowledge.T).sigmoid()
        return stu_mastery
