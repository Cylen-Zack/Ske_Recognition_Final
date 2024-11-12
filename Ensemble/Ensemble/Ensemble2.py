import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r_11 = r1[i]
        _, r_22 = r2[i]
        _, r_33 = r3[i]
        _, r_44 = r4[i]
        _, r_55 = r5[i]
        _, r_66 = r6[i]
        _, r_77 = r7[i]
        _, r_88 = r8[i]
        _, r_99 = r9[i]
        _, r_1010 = r10[i]
        _, r_1111 = r11[i]
        _, r_1212 = r12[i]
        _, r_1313 = r13[i]
        _, r_1414 = r14[i]
        _, r_1515 = r15[i]
        _, r_1616 = r16[i]
        _, r_1717 = r17[i]
        _, r_1818 = r18[i]
        _, r_1919 = r19[i]
        _, r_2020 = r20[i]
        _, r_2121 = r21[i]
        _, r_2222 = r22[i]
        _, r_2323 = r23[i]
        _, r_2424 = r24[i]
        _, r_2525 = r25[i]
        _, r_2626 = r26[i]
        _, r_2727 = r27[i]
        _, r_2828 = r28[i]
        _, r_2929 = r29[i]
        _, r_3030 = r30[i]
        _, r_3131 = r31[i]
        _, r_3232 = r32[i]
        
        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r_1010 * weights[9] \
            + r_1111 * weights[10] \
            + r_1212 * weights[11] \
            + r_1313 * weights[12] \
            + r_1414 * weights[13] \
            + r_1515 * weights[14] \
            + r_1616 * weights[15] \
            + r_1717 * weights[16] \
            + r_1818 * weights[17] \
            + r_1919 * weights[18] \
            + r_2020 * weights[19] \
            + r_2121 * weights[20] \
            + r_2222 * weights[21] \
            + r_2323 * weights[22] \
            + r_2424 * weights[23] \
            + r_2525 * weights[24] \
            + r_2626 * weights[25] \
            + r_2727 * weights[26] \
            + r_2828 * weights[27] \
            + r_2929 * weights[28] \
            + r_3030 * weights[29] \
            + r_3131 * weights[30] \
            + r_3232 * weights[31] 
        
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default = 'V1')
    
    parser.add_argument('--CTRGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/CTRGCN_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/CTRGCN_BM_3d/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/CTRGCN_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--CTRGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/CTRGCN_JM_3d/epoch1_test_score.pkl')
    
    parser.add_argument('--Mixformer_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_B_3d/epoch79_test_score.pkl')
    parser.add_argument('--Mixformer_BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_BM_3d/epoch51_test_score.pkl')
    parser.add_argument('--Mixformer_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_J_3d/epoch51_test_score.pkl')
    parser.add_argument('--Mixformer_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_JM_3d/epoch63_test_score.pkl')
    
    parser.add_argument('--Mixformer_k2_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_k2_3d/epoch68_test_score.pkl')
    parser.add_argument('--Mixformer_k2B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_k2B_3d/epoch51_test_score.pkl')
    parser.add_argument('--Mixformer_k2BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_k2BM_3d/epoch70_test_score.pkl')
    parser.add_argument('--Mixformer_k2M_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Mixformer_k2M_3d/epoch62_test_score.pkl')
    
    parser.add_argument('--MSTGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/MSTGCN_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/MSTGCN_BM_3d/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/MSTGCN_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--MSTGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/MSTGCN_JM_3d/epoch1_test_score.pkl')
    
    parser.add_argument('--TDGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TDGCN_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TDGCN_BM_3d/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TDGCN_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--TDGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TDGCN_JM_3d/epoch1_test_score.pkl')
    
    parser.add_argument('--TEGCN_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TEGCN_B_3d/epoch31_test_score.pkl')
    parser.add_argument('--TEGCN_BM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TEGCN_BM_3d/epoch36_test_score.pkl')
    parser.add_argument('--TEGCN_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TEGCN_J_3d/epoch36_test_score.pkl')
    parser.add_argument('--TEGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/TEGCN_JM_3d/epoch31_test_score.pkl')
    
    parser.add_argument('--InfoGCN_Angle_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/InfoGCN_Angle_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/InfoGCN_JM_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_k1_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/InfoGCN_k1_3d/best_score.pkl')
    parser.add_argument('--InfoGCN_loss_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/InfoGCN_loss_3d/best_score.pkl')
    
    parser.add_argument('--Sttformer_Angle_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Sttformer_Angle_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_B_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Sttformer_B_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_J_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Sttformer_J_3d/epoch1_test_score.pkl')
    parser.add_argument('--Sttformer_JM_3d', default = '/root/autodl-tmp/Ensemble/Ensemble_score/Sttformer_JM_3d/epoch1_test_score.pkl')
    arg = parser.parse_args()

    benchmark = arg.benchmark
    if benchmark == 'V1':
        npz_data = np.load('/root/autodl-tmp/ESB2.npz')
        label = npz_data['y_test']
    else:
        assert benchmark == 'V2'
        npz_data = np.load('./Model_inference/Mix_Former/dataset/save_2d_pose/V2.npz')
        label = npz_data['y_test']

    with open(arg.CTRGCN_B_3d, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.CTRGCN_BM_3d, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.CTRGCN_J_3d, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.CTRGCN_JM_3d, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.Mixformer_B_3d, 'rb') as r5:
        r5 = list(pickle.load(r5).items())
        
    with open(arg.Mixformer_BM_3d, 'rb') as r6:
        r6 = list(pickle.load(r6).items())
    
    with open(arg.Mixformer_J_3d, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(arg.Mixformer_JM_3d, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(arg.Mixformer_k2_3d, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(arg.Mixformer_k2B_3d, 'rb') as r10:
        r10 = list(pickle.load(r10).items())

    with open(arg.Mixformer_k2BM_3d, 'rb') as r11:
        r11 = list(pickle.load(r11).items())
        
    with open(arg.Mixformer_k2M_3d, 'rb') as r12:
        r12 = list(pickle.load(r12).items())
    
    with open(arg.MSTGCN_B_3d, 'rb') as r13:
        r13 = list(pickle.load(r13).items())

    with open(arg.MSTGCN_BM_3d, 'rb') as r14:
        r14 = list(pickle.load(r14).items())

    with open(arg.MSTGCN_J_3d, 'rb') as r15:
        r15 = list(pickle.load(r15).items())

    with open(arg.MSTGCN_JM_3d, 'rb') as r16:
        r16 = list(pickle.load(r16).items())

    with open(arg.TDGCN_B_3d, 'rb') as r17:
        r17 = list(pickle.load(r17).items())
        
    with open(arg.TDGCN_BM_3d, 'rb') as r18:
        r18 = list(pickle.load(r18).items())
    
    with open(arg.TDGCN_J_3d, 'rb') as r19:
        r19 = list(pickle.load(r19).items())

    with open(arg.TDGCN_JM_3d, 'rb') as r20:
        r20 = list(pickle.load(r20).items())

    with open(arg.TEGCN_B_3d, 'rb') as r21:
        r21 = list(pickle.load(r21).items())
        
    with open(arg.TEGCN_BM_3d, 'rb') as r22:
        r22 = list(pickle.load(r22).items())
    
    with open(arg.TEGCN_J_3d, 'rb') as r23:
        r23 = list(pickle.load(r23).items())

    with open(arg.TEGCN_JM_3d, 'rb') as r24:
        r24 = list(pickle.load(r24).items())
        
    with open(arg.InfoGCN_Angle_3d, 'rb') as r25:
        r25 = list(pickle.load(r25).items())
        
    with open(arg.InfoGCN_JM_3d, 'rb') as r26:
        r26 = list(pickle.load(r26).items())
    
    with open(arg.InfoGCN_k1_3d, 'rb') as r27:
        r27 = list(pickle.load(r27).items())

    with open(arg.InfoGCN_loss_3d, 'rb') as r28:
        r28 = list(pickle.load(r28).items())

    with open(arg.Sttformer_Angle_3d, 'rb') as r29:
        r29 = list(pickle.load(r29).items())
        
    with open(arg.Sttformer_B_3d, 'rb') as r30:
        r30 = list(pickle.load(r30).items())
    
    with open(arg.Sttformer_J_3d, 'rb') as r31:
        r31 = list(pickle.load(r31).items())

    with open(arg.Sttformer_JM_3d, 'rb') as r32:
        r32 = list(pickle.load(r32).items())

    space = [(0.2, 1.2) for i in range(32)]
    result = gp_minimize(objective, space, n_calls=400, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))
