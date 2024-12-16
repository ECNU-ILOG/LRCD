import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import json
import random

class DATA_SET:
    def __init__(self, train_list, test_list) -> None:
        self.train_file = train_list
        self.test_file = test_list
        
    def get_response(self):
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.test_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/response.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/response.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        data = data.to_numpy()
        return data
    
    def orcdf_response(self):
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.train_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/response.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/response.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        train_data = pd.concat(dfs, ignore_index=True)
        train_data = train_data.to_numpy()
        
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.test_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/orginal_train.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/orginal_train.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        test_data = pd.concat(dfs, ignore_index=True)
        test_data = test_data.to_numpy()
        return train_data, test_data
    
    def doa_data(self):
        _, _, stu_n, exer_n, _, _ = self.get_num()
        r_matrix = -1 * torch.ones(stu_n, exer_n)
        train_data, test_data = self.get_orginal_data()
        for _, row in test_data.iterrows():
            student_id = row[0]
            exercise_id = row[1]
            score = row[2]
            r_matrix[int(student_id), int(exercise_id)] = int(score)
        for _, row in train_data.iterrows():
            student_id = row[0]
            exercise_id = row[1]
            score = row[2]
            r_matrix[int(student_id), int(exercise_id)] = int(score)
        return r_matrix
    
    def get_num(self):
        train_stu_n, train_exer_n, total_know_n, test_stu_n, test_exer_n, train_know_n = 0, 0, 0, 0, 0, 0
        for file in self.train_file:
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                train_stu_n += df["student_num"]
                train_exer_n += df["exercise_num"]
                total_know_n += df["knowledge_num"]
                train_know_n = total_know_n
        for file in self.test_file:
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                test_stu_n += df["student_num"]
                test_exer_n += df["exercise_num"]
                total_know_n += df["knowledge_num"]
        return train_stu_n, train_exer_n, test_stu_n, test_exer_n, train_know_n, total_know_n
    
    def get_text_embedding(self, dtype, device):
        knowledge_emb, student_emb, exercise_emb = [], [], []
        for file in self.train_file:
            know_emb = pd.read_csv(file + "/know_embedding.csv", header=None)
            knowledge_emb.append(torch.tensor(know_emb.values, dtype=dtype, device=device))
            stu_emb = pd.read_csv(file + "/stu_embedding.csv", header=None)
            student_emb.append(torch.tensor(stu_emb.values, dtype=dtype, device=device))
            exer_emb = pd.read_csv(file + "/exer_embedding.csv", header=None)
            exercise_emb.append(torch.tensor(exer_emb.values, dtype=dtype, device=device))
        total_student_emb = torch.cat(student_emb, dim=0)
        total_exercise_emb = torch.cat(exercise_emb, dim=0)
        
        student_emb, exercise_emb = [], []
        for file in self.test_file:
            know_emb = pd.read_csv(file + "/know_embedding.csv", header=None)
            knowledge_emb.append(torch.tensor(know_emb.values, dtype=dtype, device=device))
            stu_emb = pd.read_csv(file + "/stu_embed_test.csv", header=None)
            student_emb.append(torch.tensor(stu_emb.values, dtype=dtype, device=device))
            exer_emb = pd.read_csv(file + "/exer_embed_test.csv", header=None)
            exercise_emb.append(torch.tensor(exer_emb.values, dtype=dtype, device=device))
        knowledge_emb = torch.cat(knowledge_emb, dim=0)
        student_emb = torch.cat(student_emb, dim=0)
        exercise_emb = torch.cat(exercise_emb, dim=0)
        train_data_emb = {
            "student": total_student_emb,
            "exercise": total_exercise_emb,
            "disc": total_exercise_emb,
            "knowledge": knowledge_emb
        }
        test_data_emb = {
            "student": student_emb,
            "exercise": exercise_emb,
            "disc": exercise_emb,
            "knowledge": knowledge_emb
        }
        return train_data_emb, test_data_emb
    
    def get_data(self):
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.train_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/response.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/response.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        train_data = pd.concat(dfs, ignore_index=True)
        
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.test_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/test_response.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/test_response.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        test_data = pd.concat(dfs, ignore_index=True)
        return train_data, test_data
    
    def get_orginal_data(self):
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.test_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/orginal_train.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/orginal_train.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        train_data = pd.concat(dfs, ignore_index=True)
        
        dfs = []
        add_stu_num, add_exer_num = 0, 0
        for index, file in enumerate(self.test_file):
            if index == 0:
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num, add_exer_num = json_f["student_num"], json_f["exercise_num"]
                df = pd.read_csv(file + "/test_response.csv", header=None)
                dfs.append(df)
            else:
                df = pd.read_csv(file + "/test_response.csv", header=None)
                df.iloc[:, 0] += add_stu_num
                df.iloc[:, 1] += add_exer_num
                with open(file + "/data.json", 'r') as json_file:
                    json_f = json.load(json_file)
                add_stu_num += json_f["student_num"]
                add_exer_num += json_f["exercise_num"]
                dfs.append(df)
        test_data = pd.concat(dfs, ignore_index=True)
        return train_data, test_data
    
    def get_q_matrix(self):
        _, train_exer_n, _, test_exer_n, _, total_know_n = self.get_num()
        train_q_matrix = torch.zeros(size=(train_exer_n, total_know_n))
        test_q_matrix = torch.zeros(size=(test_exer_n, total_know_n))
        tr_exer_n, te_exer_n, know_n = 0, 0, 0
        for file in self.train_file:
            qf = pd.read_csv(file + "/q_matrix.csv", header=None)
            q_matrix = torch.tensor(qf.values)
            exer, know = q_matrix.shape
            train_q_matrix[tr_exer_n:tr_exer_n + exer, know_n:know_n + know] = q_matrix
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                tr_exer_n += df["exercise_num"]
                know_n += df["knowledge_num"]
        for file in self.test_file:
            qf = pd.read_csv(file + "/q_matrix.csv", header=None)
            q_matrix = torch.tensor(qf.values)
            exer, know = q_matrix.shape
            test_q_matrix[te_exer_n:te_exer_n + exer, know_n:know_n + know] = q_matrix
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                te_exer_n += df["exercise_num"]
                know_n += df["knowledge_num"]
        return train_q_matrix, test_q_matrix
    
    def get_iter(self, batch_size):
        train_q_matrix, test_q_matrix = self.get_q_matrix()
        train_data, test_data = self.get_data()
        user = train_data.iloc[:, 0]
        item = train_data.iloc[:, 1]
        score = train_data.iloc[:, 2]
        train_dataset = TensorDataset(
            torch.tensor(user.apply(int), dtype=torch.int64),
            torch.tensor(item.apply(int), dtype=torch.int64),
            train_q_matrix[np.array(item, dtype=int), :],
            torch.tensor(score, dtype=torch.float64)
        )
        user = test_data.iloc[:, 0]
        item = test_data.iloc[:, 1]
        score = test_data.iloc[:, 2]
        test_dataset = TensorDataset(
            torch.tensor(user.apply(int), dtype=torch.int64),
            torch.tensor(item.apply(int), dtype=torch.int64),
            test_q_matrix[np.array(item, dtype=int), :],
            torch.tensor(score, dtype=torch.float64)
        )
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
