import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score, accuracy_score
from extract.orcdf import ORCDF_Extractor
from interact.kancd import KANCD_IF
from interact.kscd import KSCD_IF
from base_model import base_CD

class orcdf_model(base_CD):
    def __init__(self, dataset, **kwargs):
        super(orcdf_model, self).__init__()
        self.dataset = dataset
        self.flip_ratio = kwargs['flip_ratio']  
        self.inter = kwargs['inter'] 
        self.orcdf_mode = kwargs['orcdf_mode'] 
        self.device = kwargs['device'] 
        _, _, _, _, tr_know_n, to_know_n = dataset.get_num()
        _, q_matrix = dataset.get_q_matrix()

        self.know_n = to_know_n
        
        self.doa_dict = {
            "r_matrix": dataset.doa_data(),
            "data": dataset.get_response(),
            "q_matrix": q_matrix,
            "know_n": tr_know_n
        }
        
        self.extractor = ORCDF_Extractor(dataset, **kwargs)
        if self.inter == 'kancd':
            self.interfunc = KANCD_IF(self.know_n, kwargs['latent_dim'], kwargs['dropout'], kwargs['device'], kwargs['dtype'])
        elif self.inter == 'kscd':
            self.interfunc = KSCD_IF(self.know_n, kwargs['latent_dim'], kwargs['dropout'], kwargs['device'], kwargs['dtype'])
    
    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)
    
    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()
    
    def __create_adj_se(self, np_response, flag, is_subgraph=False):
        if flag == 'train':
            self.student_num, self.exercise_num, _, _, _, _ = self.dataset.get_num()
        else:
            _, _, self.student_num, self.exercise_num, _, _ = self.dataset.get_num()
        
        if is_subgraph:
            if self.orcdf_mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.orcdf_mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek, flag):
        if flag == 'train':
            self.student_num, self.exercise_num, _, _, _, self.knowledge_num = self.dataset.get_num()
        else:
            _, _, self.student_num, self.exercise_num, _, self.knowledge_num = self.dataset.get_num()
        
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num:se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)
    
    def train_model(self, batch_size, epoch_num, lr, device):
        tr_res, te_res = self.dataset.orcdf_response()
        
        train_loader, test_loader = self.dataset.get_iter(batch_size)
        ek_graph1, ek_graph2 = self.dataset.get_q_matrix()

        se_graph_right1, se_graph_wrong1 = [self.__create_adj_se(tr_res, 'train', is_subgraph=True)[i] for i in range(2)]
        se_graph1 = self.__create_adj_se(tr_res, 'train', is_subgraph=False)

        se_graph_right2, se_graph_wrong2 = [self.__create_adj_se(te_res, 'test', is_subgraph=True)[i] for i in range(2)]
        se_graph2 = self.__create_adj_se(te_res, 'test', is_subgraph=False)

        if self.flip_ratio:
            def get_flip_data():
                np_response_flip = tr_res
                column = np_response_flip[:, 2]
                probability = np.random.choice([True, False], size=column.shape,
                                               p=[self.flip_ratio, 1 - self.flip_ratio])
                column[probability] = 1 - column[probability]
                np_response_flip[:, 2] = column
                return np_response_flip

        graph_dict_train = {
            'right': self.__final_graph(se_graph_right1, ek_graph1, 'train'),
            'wrong': self.__final_graph(se_graph_wrong1, ek_graph1, 'train'),
            'response': tr_res,
            'Q_Matrix': ek_graph1,
            'flip_ratio': self.flip_ratio,
            'all': self.__final_graph(se_graph1, ek_graph1, 'train'),
        }
        graph_dict_test = {
            'right': self.__final_graph(se_graph_right2, ek_graph2, 'test'),
            'wrong': self.__final_graph(se_graph_wrong2, ek_graph2, 'test'),
            'response': te_res,
            'Q_Matrix': ek_graph2,
            'flip_ratio': self.flip_ratio,
            'all': self.__final_graph(se_graph2, ek_graph2, 'test'),
        }

        for i in range(epoch_num):
            print("[epoch %d]:" % (i))
            self.extractor.get_graph_dict(graph_dict_train)
            self.extractor.get_flip_graph('train')
            self.train(lr, self.extractor, self.interfunc, train_loader, device)
            self.extractor.get_graph_dict(graph_dict_test)
            self.extractor.get_flip_graph('test')
            self.eval(self.extractor, self.interfunc, test_loader, device, self.doa_dict)
