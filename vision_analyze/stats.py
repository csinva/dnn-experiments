import numpy as np

# simple pearson correlation
def corr(vec1, vec2):
    return np.corrcoef(vec1.flatten(), vec2.flatten())[0, 1]

# calc mean correlation of columns of mat1 and mat
def calc_mean_corr(mat1, mat2):
    corrs = []
    for c in range(mat1.shape[1]):
        corrs.append(np.corrcoef(mat1[:, c], mat2[:, c])[0, 1])
    return np.mean(corrs)


# predict on X_train and X_test and return accs
def calc_accs(model, X_train, Y_train, X_test, Y_test):
    preds = model(X_train).data.cpu().numpy().argmax(axis=1)
    accs_train = preds==Y_train
    preds = model(X_test).data.cpu().numpy().argmax(axis=1)
    accs_test = preds==Y_test
    return np.mean(accs_train), np.mean(accs_test)

def calc_corrs(p1, p2, labs):
    p1_max = np.argmax(p1, axis=1)
    p2_max = np.argmax(p2, axis=1)
    idxs_both_correct = (p1_max == labs) * (p2_max == labs)

    # try with softmax also

    # raw correlation
    raw_corr = stats.corr(p1.flatten(), p2.flatten())
#     print(f'raw corr: {raw_corr: 0.5f}')

    # correlation within each class for all columns
    class_corrs = np.zeros(1000)
    for lab_num in range(1000):
        idxs = labs == lab_num
        class_corrs[lab_num] = stats.corr(p1[idxs].flatten(), p2[idxs].flatten())
    intra_class_corr_all = np.mean(class_corrs)
#     print(f'intra-class corr all columns: {intra_class_corr_all: 0.5f}')

    # correlation within each class for just correct column
    class_corrs = np.zeros(1000)
    for lab_num in range(1000):
        idxs = labs == lab_num
        class_corrs[lab_num] = stats.corr(p1[idxs][:, lab_num], p2[idxs][:, lab_num])
    intra_class_corr_column = np.mean(class_corrs)    
#     print(f'\tintra-class corr class-column: {np.mean(intra_class_corr_correct): 0.5f}')

    # correlation when both networks correct
    class_corrs = np.zeros(1000)
    for lab_num in range(1000):
        idxs = (labs == lab_num) * (idxs_both_correct)
        class_corrs[lab_num] = stats.corr(p1[idxs].flatten(), p2[idxs].flatten())
    intra_class_corr_correct = np.mean(class_corrs)
#     print(f'intra-class corr for correct points: {intra_class_corr_correct: 0.5f}')

    # rank correlation
    rank_corrs = np.zeros(1000)
    for lab_num in range(1000):
        idxs = (labs == lab_num)
        rank_corrs[lab_num] = scipy.stats.spearmanr(p1[idxs].flatten(), p2[idxs].flatten())[0]
    intra_class_rank_corr = np.mean(rank_corrs)
    
    
    return {'raw_corr': raw_corr, 
            'intra_class_corr_all': intra_class_corr_all, 
            'intra_class_corr_correct': intra_class_corr_correct, 
            'intra_class_corr_column': intra_class_corr_column,
            'intra_class_rank_corr': intra_class_rank_corr}

# 2 inputs: k x num_classes 
# output: whether top-1 in first is in top-k of 2nd
# note: this is not symmetric
# note: max is last
def num_agree_topk(inds1, inds2, k=None):
    if k is None:
        k = inds1.shape[0]
    top_inds1 = inds1[-1]
    num_agree = 0
    for class_num in range(inds1.shape[1]):
        if top_inds1[class_num] in inds2[-1 * k:, class_num]:
            num_agree += 1
    return num_agree