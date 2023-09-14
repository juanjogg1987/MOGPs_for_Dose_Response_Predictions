import numpy as np
import matplotlib.pyplot as plt
import GPy

from scipy.linalg import cholesky, cho_solve
from GPy.util import linalg
from sklearn import metrics
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate

def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

def ArrangeData_to_CrossVal(Ntasks, Nfold, nsplits, list_folds, Xall, Yall, Emax_all, IC50_all, AUC_all):
    if Nfold < nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        Xtrain_aux = Xall[train_ind].copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:, None]
        Ytrain = Yall[train_ind].T.flatten().copy()[:, None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()
    else:
        print(f"Train ovell all Data")
        _, test_ind = list_folds[
            0]  # Just assigning by defaul fold0 as the test (of course not to report it as a result)
        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        Xtrain_aux = Xall.copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:, None]
        Ytrain = Yall.T.flatten().copy()[:, None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()
    return Xtrain, Ytrain, Xval, Xval_aux, Yval, Emax_val, IC50_val, AUC_val

def Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose):
    x_lin_tile = np.tile(x_lin, (params_4_sig_train.shape[0], 1))
    # (x_lin,params_4_sig_train.shape[0],1).shape
    Ydose_res = []
    AUC = []
    IC50 = []
    Ydose50 = []
    Emax = []
    for i in range(params_4_sig_train.shape[0]):
        Ydose_res.append(sigmoid_4_param(x_lin_tile[i, :], *params_4_sig_train[i, :]))
        AUC.append(metrics.auc(x_lin_tile[i, :], Ydose_res[i]))
        Emax.append(Ydose_res[i][-1])
        res1 = (Ydose_res[i] < 0.507)
        res2 = (Ydose_res[i] > 0.493)
        if (res1 & res2).sum() > 0:
            Ydose50.append(Ydose_res[i][res1 & res2].mean())
            IC50.append(x_lin[res1 & res2].mean())
        elif Ydose_res[i][-1]<0.5:
           for dose_j in range(x_lin.shape[0]):
               if(Ydose_res[i][dose_j] < 0.5):
                   break
           Ydose50.append(Ydose_res[i][dose_j])
           aux_IC50 = x_lin[dose_j]  #it has to be a float not an array to avoid bug
           IC50.append(aux_IC50)
        else:
            Ydose50.append(0.5)
            IC50.append(1.5)

    return Ydose50,Ydose_res,IC50,AUC,Emax

def Predict_Curve_and_SummaryMetrics(x_dose,x_dose_new, model = None, Xval = None,m_pred_curve = None, v_pred=None,MOGPsamples = 0):
    if m_pred_curve is None and v_pred is None:
        Ntasks = model.mul.coregion.W.shape[0]
        if MOGPsamples == 0:
            m_pred, v_pred = model.predict(Xval, full_cov=False)
            #v_pred_curve = v_pred.reshape(Ntasks, Xval.shape[0] // Ntasks).T.copy()
            v_pred = np.diag(np.diag(v_pred))
        else:
            m_pred, v_pred = model.predict(Xval, full_cov=True)
            #print("INININI")
            #v_pred_curve = v_pred.reshape(Ntasks, Xval.shape[0] // Ntasks).T.copy()
        m_pred_curve = m_pred.reshape(Ntasks, Xval.shape[0] // Ntasks).T.copy()
    else:
        v_pred = np.diag(v_pred.flatten())
        #print(v_pred.shape)

    #x_dose_new = np.linspace(0.111111, 1.0, 1000)
    Ydose50_pred = []
    IC50_pred = []
    AUC_pred = []
    Emax_pred = []
    Y_pred_interp_all = []
    std_upper_interp_all = []
    std_lower_interp_all = []
    y_interp_sampled = []
    y_interp_sampled_all = []
    Ncurves = m_pred_curve.shape[0]
    for i in range(Ncurves):
        y_resp = m_pred_curve[i, :].copy()
        #std_upper = y_resp + 2 * np.sqrt(v_pred_curve[i, :])
        #std_lower = y_resp - 2 * np.sqrt(v_pred_curve[i, :])
        std_upper = y_resp + 2 * np.sqrt(np.diag(v_pred[i::Ncurves, i::Ncurves]))
        std_lower = y_resp - 2 * np.sqrt(np.diag(v_pred[i::Ncurves, i::Ncurves]))
        f = interp1d(x_dose, y_resp)
        # We use a Pchip interpolation method
        y_resp_interp = pchip_interpolate(x_dose, y_resp, x_dose_new)
        std_upper_interp = pchip_interpolate(x_dose, std_upper, x_dose_new)
        std_lower_interp = pchip_interpolate(x_dose, std_lower, x_dose_new)

        "Get MOGP samples interpolated"
        if MOGPsamples>0:
            y_interp_sampled = []
            for k in range(MOGPsamples):
                y_sampled = np.random.multivariate_normal(y_resp.flatten(),v_pred[i::Ncurves, i::Ncurves])
                y_interp_sampled.append(pchip_interpolate(x_dose, y_sampled, x_dose_new))

        Y_pred_interp_all.append(y_resp_interp)
        y_interp_sampled_all.append(y_interp_sampled)
        std_upper_interp_all.append(std_upper_interp)
        std_lower_interp_all.append(std_lower_interp)
        AUC_pred.append(metrics.auc(x_dose_new, y_resp_interp))
        Emax_pred.append(y_resp_interp[-1])

        res1 = y_resp_interp < 0.507
        res2 = y_resp_interp > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum() > 0:
            res_IC50 = np.arange(res_aux[0], res_aux[0] + res_aux.shape[0]) == res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred.append(y_resp_interp[res_aux].mean())
            IC50_pred.append(x_dose_new[res_aux].mean())
        elif y_resp_interp[-1] < 0.5:
            for dose_j in range(x_dose_new.shape[0]):
                if (y_resp_interp[dose_j] < 0.5):
                    break
            Ydose50_pred.append(y_resp_interp[dose_j])
            aux_IC50 = x_dose_new[dose_j]  # it has to be a float not an array to avoid bug
            IC50_pred.append(aux_IC50)
        else:
            Ydose50_pred.append(0.5)
            IC50_pred.append(1.5)
    "TODO: make this function return InterpolGP samples"
    return np.array(Y_pred_interp_all),np.array(std_upper_interp_all),np.array(std_lower_interp_all), y_interp_sampled_all,Ydose50_pred,IC50_pred, AUC_pred, Emax_pred

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

