import numpy as np
import matplotlib.pyplot as plt
import GPy

from scipy.linalg import cholesky, cho_solve
from GPy.util import linalg

# Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        L1 = np.diag(np.sqrt(np.diag(V1)))
        L2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        L1 = cholesky(V1, lower=True)
        L2 = cholesky(V2, lower=True)
        V2_inv, _ = linalg.dpotri(np.asfortranarray(L2))
        # V2_inv = np.linalg.inv(V2)
    # print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    KL = np.clip(KL, 0.0, 1.7e308)  # This is to avoid negative values
    return KL  # This is to avoid any negative due to numerical instability

def KLD_Gaussian_NoChol(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # print("shape m1", m1.shape)
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        V1 = np.diag(np.sqrt(np.diag(V1)))
        V2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        # V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        V2_inv = np.linalg.inv(V2)
    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    KL = np.clip(KL,0.0,1.7e308)    #This is to avoid negative values
    return KL  # This is to avoid any negative due to numerical instability

def KLRelevance_MOGP_GPy(train_x, model, delta,which_p,diag = False,Use_Cholesky = False, ToSave = False, FileName = "Melanoma"):
    Ntasks = model.kern.coregion.B.shape[0]
    Nall,P = train_x.shape
    Ntasks = model.kern.coregion.B.shape[0]
    N = Nall//Ntasks
    P = P-1 #Here we do not have into account the last feature since it represents a label to the output
    relevance = np.zeros((N, P))
    # delta = 1.0e-4
    jitter = 1.0e-15

    #which_p = int(config.feature)
    print(f"Analysing Feature {which_p} of {P}...")
    for p in range(which_p,which_p+1):
        for n in range(N):
            #ind_all_tasks = np.array([n,n+N,n+N*2,n+N*3,n+N*4,n+N*5,n+N*6])
            ind_all_tasks = np.array([n + i * N for i in range(0, Ntasks)])
            x_plus = train_x[ind_all_tasks, :].copy()
            x_minus = train_x[ind_all_tasks, :].copy()
            x_plus[:, p] = x_plus[:, p] + delta
            x_minus[:, p] = x_minus[:, p] - delta

            m1, V1 = model.predict(train_x[ind_all_tasks,:],full_cov = True)  # predict_xn.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2, V2 = model.predict(x_plus,full_cov = True)  # predict_xn_delta.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2_minus, V2_minus = model.predict(x_minus,full_cov = True)  # predict_xn_delta_min.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D

            use_diag = diag
            if Use_Cholesky:
                KL_plus = np.sqrt(2.0*KLD_Gaussian(m1, V1, m2, V2,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(2.0*KLD_Gaussian(m1, V1, m2_minus, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            else:
                KL_plus = np.sqrt(2.0*KLD_Gaussian_NoChol(m1, V1, m2, V2,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(2.0*KLD_Gaussian_NoChol(m1, V1, m2_minus, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2

            relevance[n, p] = 0.5 * (KL_plus + KL_minus) / delta

    "Correction of Nan data in the relevance"
    NonNan_ind = np.where(~np.isnan(relevance[:,which_p]))
    Nan_ind = np.where(np.isnan(relevance[:,which_p]))[0]
    relevance[Nan_ind,which_p]= (relevance[NonNan_ind,which_p].max()-relevance[NonNan_ind,which_p].min())/2.0
    #print(f"Relevance of Features:\n {np.mean(relevance,0)}")
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Below we save the relevance of each n-th data regarding the p-th feature"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if ToSave:
        f= open("Relevance_"+FileName+".txt","a+")
        f.write(f"{which_p}")
        for n in range(N):
            f.write(",")
            f.write(f"{relevance[n,which_p]:0.5}")
        f.write(f"\n")
        f.close()
    return relevance

