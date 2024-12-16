import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import Parallel, delayed

def top_k_concepts(top_k, q_matrix, tmp_set):
    arr = np.array(tmp_set[:, 1], dtype=int)
    counts = np.sum(q_matrix[np.array(tmp_set[:, 1], dtype=int), :], axis=0)
    return np.argsort(counts).tolist()[:-top_k - 1:-1]

def __calculate_doa_k(mas_level, q_matrix, r_matrix, k):
    n_questions, _ = q_matrix.shape
    stu, exer = r_matrix.shape
    numerator = 0
    denominator = 0
    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * column_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)
        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k


def __calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        mas_level_block = mas_level[start:end, :]
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k

def degree_of_agreement(mastery_level, doa_list):
    r_matrix = doa_list['r_matrix'].numpy()
    data = doa_list['data']
    q_matrix = doa_list['q_matrix'].numpy()
    know_n = q_matrix.shape[1]
    know_n -= doa_list['know_n']
    if know_n > 30:
        concepts = top_k_concepts(10, q_matrix, data)
        doa_k_list = Parallel(n_jobs=-1)(
            delayed(__calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    else:
        doa_k_list = Parallel(n_jobs=-1)(
            delayed(__calculate_doa_k)(mastery_level, q_matrix, r_matrix, k) for k in range(doa_list['know_n'], know_n + doa_list['know_n']))
    doa_k_list = [x for x in doa_k_list if x != 0]
    return np.mean(doa_k_list)

class base_CD(nn.Module):
    def __init__(self):
        super(base_CD, self).__init__()

    def train(self, learning_rate, extractor, inter_func, dataloader, device):
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': extractor.parameters(),
                                 'lr': learning_rate},
                                {'params': inter_func.parameters(),
                                 'lr': learning_rate}])
        extractor.train()
        inter_func.train()
        epoch_loss = []
        for batch_data in dataloader:
            student_id, exercise_id, q_mask, r = batch_data
            student_id: torch.Tensor = student_id.to(device)
            exercise_id: torch.Tensor = exercise_id.to(device)
            q_mask: torch.Tensor = q_mask.to(device)
            r: torch.Tensor = r.to(device)
            _ = extractor.extract(student_id, exercise_id, 'train')
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred_r = inter_func.compute(student_ts, diff_ts, disc_ts, knowledge_ts, q_mask)
            if len(_) > 4:
                extra_loss = _[4].get('extra_loss', 0)
            else:
                extra_loss = 0
            loss = loss_func(pred_r, r) + extra_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.mean()))
        print("average train loss: %.6f" % (float(np.mean(epoch_loss))))
        wandb.log({"train_loss": float(np.mean(epoch_loss))})

    def eval(self, extractor, inter_func, dataloader, device, doa_list):
        loss_func = nn.BCELoss()
        epoch_loss, auc_list, acc_list = [], [], []
        y_pred, y_true = [], []
        for batch_data in dataloader:
            extractor.eval()
            inter_func.eval()
            with torch.no_grad():
                student_id, exercise_id, q_mask, r = batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                r: torch.Tensor = r.to(device)
                _ = extractor.extract(student_id, exercise_id, 'test')
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                pred_r = inter_func.compute(student_ts, diff_ts, disc_ts, knowledge_ts, q_mask)
                if len(_) > 4:
                    extra_loss = _[4].get('extra_loss', 0)
                else:
                    extra_loss = 0
                l = loss_func(pred_r, r) + extra_loss
                y_pred.extend(pred_r.detach().cpu().tolist())
                y_true.extend(r.tolist())
                epoch_loss.append(float(l.mean()))

        auc_list.append(roc_auc_score(y_true, y_pred))
        acc_list.append(accuracy_score(y_true, np.array(y_pred) >= 0.5))
        stu_mas = inter_func.transform(extractor.update("student"), extractor.update("knowledge"))
        loss_value = float(np.mean(epoch_loss))
        auc_value = round(float(np.mean(auc_list)) * 100, 2)
        acc_value = round(float(np.mean(acc_list)) * 100, 2)
        doa_value = round(degree_of_agreement(stu_mas.detach().cpu().numpy(), doa_list) * 100, 2)
        print("average test loss: %.6f" % (loss_value))
        test_dict = {
            "test_loss": loss_value,
            "auc": auc_value,
            "acc": acc_value,
            "doa": doa_value
        }
        print("auc: %.2f , acc: %.2f , doa: %.2f" % (auc_value, acc_value, doa_value))
        wandb.log(test_dict)
