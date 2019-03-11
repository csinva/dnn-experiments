import numpy as np

# simple pearson correlation
def corr(vec1, vec2):
    return np.corrcoef(vec1, vec2)[0, 1]

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