from extract.default import extract_fea
from interact.kscd import KSCD_IF
from base_model import base_CD

class kscd_model(base_CD):
    def __init__(self, dataset, **kwargs):
        super(kscd_model, self).__init__()
        self.dataset = dataset
        _, _, _, _, tr_know_n, to_know_n = dataset.get_num()
        _, q_matrix = dataset.get_q_matrix()
        self.know_n = to_know_n
        self.doa_dict = {
            "r_matrix": dataset.doa_data(),
            "data": dataset.get_response(),
            "q_matrix": q_matrix,
            "know_n": tr_know_n
        }
        
        self.extractor = extract_fea(dataset, kwargs['latent_dim'], kwargs['dtype'], kwargs['device'])
        self.interfunc = KSCD_IF(self.know_n, kwargs['latent_dim'], kwargs['dropout'], kwargs['device'], kwargs['dtype'])
        
    def train_model(self, batch_size, epoch_num, lr, device):
        train_loader, test_loader = self.dataset.get_iter(batch_size)
        for i in range(epoch_num):
            print("[epoch %d]:" % (i))
            self.train(lr, self.extractor, self.interfunc, train_loader, device)
            self.eval(self.extractor, self.interfunc, test_loader, device, self.doa_dict)
