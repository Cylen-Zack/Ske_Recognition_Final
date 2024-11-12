import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument('--benchmark', default = 'V1')
    
    parser.add_argument('--CTRGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/CTRGCN_B_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/CTRGCN_BM_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/CTRGCN_J_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/CTRGCN_JM_3d_Test/epoch1_test_score.pkl')
    
    parser.add_argument('--Mixformer_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_B_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_BM_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_J_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_JM_3d_Test/epoch1_test_score.pkl')
    
    parser.add_argument('--Mixformer_k2_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_k2_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_k2B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_k2B_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_k2BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_k2BM_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--Mixformer_k2M_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Mixformer_k2M_3d_Test/epoch1_test_score.pkl')
    
    parser.add_argument('--MSTGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/MSTGCN_B_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/MSTGCN_BM_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/MSTGCN_J_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/MSTGCN_JM_3d_Test/epoch1_test_score.pkl')
    
    parser.add_argument('--TDGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/TDGCN_B_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/TDGCN_BM_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/TDGCN_J_3d_Test/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/TDGCN_JM_3d_Test/epoch1_test_score.pkl')
    
    parser.add_argument('--TEGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Test_TEGCN_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--TEGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Test_TEGCN_BM_3d/epoch1_test_score.pkl')
    parser.add_argument('--TEGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Test_TEGCN_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--TEGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Test_TEGCN_JM_3d/epoch1_test_score.pkl')
    
    parser.add_argument('--InfoGCN_Angle_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/InfoGCN_Angel_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/InfoGCN_JM_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_k1_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/InfoGCN_k1_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_loss_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/InfoGCN_loss_3d/best_score.pkl')
    
    parser.add_argument('--Sttformer_Angle_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Sttformer_Angle_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_B_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Sttformer_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_J_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Sttformer_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_JM_3d', default = '/root/autodl-tmp/Ensemble/Test_Score/Sttformer_JM_3d/epoch1_test_score.pkl')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name.split('A')[1][:3])
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    r1 = args.CTRGCN_B_3d
    r2 = args.CTRGCN_BM_3d
    r3 = args.CTRGCN_J_3d
    r4 = args.CTRGCN_JM_3d
    
    r5 = args.Mixformer_B_3d
    r6 = args.Mixformer_BM_3d
    r7 = args.Mixformer_J_3d
    r8 = args.Mixformer_JM_3d
    
    r9 = args.Mixformer_k2_3d
    r10 = args.Mixformer_k2B_3d
    r11 = args.Mixformer_k2BM_3d
    r12 = args.Mixformer_k2M_3d
    
    r13 = args.MSTGCN_B_3d
    r14 = args.MSTGCN_BM_3d
    r15 = args.MSTGCN_J_3d
    r16 = args.MSTGCN_JM_3d
    
    r17 = args.TDGCN_B_3d
    r18 = args.TDGCN_BM_3d
    r19 = args.TDGCN_J_3d
    r20 = args.TDGCN_JM_3d
    
    r21 = args.TEGCN_B_3d
    r22 = args.TEGCN_BM_3d
    r23 = args.TEGCN_J_3d
    r24 = args.TEGCN_JM_3d
    
    r25 = args.InfoGCN_Angle_3d
    r26 = args.InfoGCN_JM_3d
    r27 = args.InfoGCN_k1_3d
    r28 = args.InfoGCN_loss_3d
    
    r29 = args.Sttformer_Angle_3d
    r30 = args.Sttformer_B_3d
    r31 = args.Sttformer_J_3d
    r32 = args.Sttformer_JM_3d


    File = [ r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32]    
    if args.benchmark == 'V1':
        Numclass = 155
        Sample_Num = 4307
        Rate = [1.2, 0.2, 1.2, 1.2, 0.5869285335030839, 1.2, 0.2, 0.2, 0.2, 0.2, 0.5172621544341824, 0.8175532182103975, 0.2, 0.9287456581275864, 0.5527987293353862, 0.2, 0.2, 0.2, 0.2, 0.2, 1.2, 0.6705282434099138, 0.2160046006484522, 1.0751951191401607, 0.8617408724044917, 1.2, 0.2, 1.002670878417916, 1.2, 0.2, 1.2, 0.6737282840626437]
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        print(final_score.shape)
        np.save(f'/root/autodl-tmp/Final_result/pred.npy', final_score)
        # npz_data = np.load('/root/autodl-tmp/ICMEW2024-Track10/Model_inference/Mix_GCN/dataset/2d_pose/Valid_data.npz')
        # true_label = npz_data['y_test']
    
    if args.benchmark == 'V2':
        Numclass = 155
        Sample_Num = 6599
        Rate = [0.7214280414594167, 1.2, 0.2, 1.2, 1.2, 0.9495413913063555, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 0.6745433985952421, 0.3926448734729191, 0.2, 0.2]  
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label(val_txt_file)
    
    # Acc = Cal_Acc(final_score, true_label)

    # print('acc:', Acc)
