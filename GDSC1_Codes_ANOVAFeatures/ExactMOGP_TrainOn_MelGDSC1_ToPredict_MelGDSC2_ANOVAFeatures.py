import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

plt.close('all')

#_FOLDER = "/home/ac1jjgg/Dataset_BRAF_NoReplica_ANOVA_Features/GDSC1/"
#_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_BRAF_NoReplica_ANOVA_Features/GDSC1/"
_FOLDER = "../Dataset_BRAF_NoReplica_ANOVA_Features/GDSC1/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys
sys.path.append('..')

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
"Best Model Drug1061: ExactMOGP_and_KLRelevance_MelanomaGDSC1_ANOVAFeatures.py -i 30 -s 1.0000 -k 2 -w 1.0000 -r 1014 -p 505 -d 1061 -e 0"
"Best Model Drug1373: ExactMOGP_and_KLRelevance_MelanomaGDSC1_ANOVAFeatures.py -i 30 -s 1.0000 -k 2 -w 1.0000 -r 1013 -p 221 -d 1373 -e 0"
"Best Model Drug1036: ExactMOGP_and_KLRelevance_MelanomaGDSC1_ANOVAFeatures.py -i 30 -s 3.0000 -k 2 -w 0.0100 -r 1018 -p 946 -d 1036 -e 0"

print("TODO: check that Niter is actually used!!!!")
class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:d:e:')
        # opts = dict(opts)
        # print(opts)
        "TODO: check that Niter is actually used!!!!"
        self.N_iter = 30    #number of iterations
        self.which_seed = 1014    #change seed to initialise the hyper-parameters
        self.rank = 2
        self.scale = 1.0
        self.weight = 2.0
        self.bash = "946"
        self.drug_name = "1036"
        self.feature = 2

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-k':  # ran(k)
                self.rank = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-w':
                self.weight = arg
            if op == '-d':
                self.drug_name = arg
            if op == '-e':
                self.feature = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"#Bear in mind that the drugs 1061 and 1036 have 9 concentrations, i.e., d1, d2, d3, d4, d5, d6, d7, d8 and d9 "
"#but the drugs 1371 and 1373 have 5 concentrations, i.e, d1, d3, d5, d7 and d9"
"#Drugs 1036 and 1371 are the same one, but the former has 9 concentrations and the latter 5 concentrations"

dconcentr = {"1061": "9conc","1036": "9conc","1373": "5conc","1371": "5conc"}
feat_for_drug = {"1061": "2-fold","1036": "2-fold","1373": "4-fold","1371": "4-fold"}
name_for_KLrelevance = 'GDSC1_melanoma_BRAF_'+dconcentr[config.drug_name]+'_noreps_v3.csv'
name_ANOVA_feat_file = 'GDSC1_BRAFmelanoma_ANOVAfeatures_'+feat_for_drug[config.drug_name]+'.csv'

df_train_No_MolecForm = pd.read_csv(_FOLDER + name_for_KLrelevance)  # Contain Train dataset prepared by Subhashini-Evelyn
df_ANOVA_feat_Names = pd.read_csv(_FOLDER + name_ANOVA_feat_file)  # Contain Feature Names used by ANOVA
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_No_MolecForm = df_train_No_MolecForm[(df_train_No_MolecForm["DRUG_ID"]==int(config.drug_name))]
try:
    df_train_No_MolecForm = df_train_No_MolecForm.drop(columns='Drug_Name')
except:
    pass

# the column index 29 for Drug 1061 the input features start
# but column index 21 is for Drug 1373
if config.drug_name == "1061" or config.drug_name == "1036":
    Dnorm_cell = 9
    start_pos_features = 29
elif config.drug_name == "1373" or config.drug_name == "1371":
    Dnorm_cell = 5
    start_pos_features = 21

print(df_train_No_MolecForm.columns[start_pos_features])

#print("Columns with std equal zero:")
#print("Number of columns with zero std:", np.sum(df_train_No_MolecForm.std(0) == 0.0))
#print(np.where(df_train_No_MolecForm.std(0) == 0.0))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Scale the Genomic features to be between 0 and 1"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scaler = MinMaxScaler().fit(df_train_No_MolecForm[df_train_No_MolecForm.columns[start_pos_features:]])
X_train_features = scaler.transform(df_train_No_MolecForm[df_train_No_MolecForm.columns[start_pos_features:]])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we extract from the dataframe the D outputs related to each dose concentration"
"Below we select 9 concentration since GDSC1 has that number for Drugs 1036 and 1061"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
y_train_drug = np.clip(df_train_No_MolecForm["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug.shape)
for i in range(2, Dnorm_cell+1):
    y_train_drug = np.concatenate(
        (y_train_drug, np.clip(df_train_No_MolecForm["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", y_train_drug.shape)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Since we fitted the dose response curves with a Sigmoid4_parameters function"
"We extract the optimal coefficients in order to reproduce such a Sigmoid4_parameters fitting"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train = df_train_No_MolecForm["param_" + str(1)].values[:, None]
for i in range(2, 5):  #here there are four params for sigmoid4
    params_4_sig_train = np.concatenate(
        (params_4_sig_train, df_train_No_MolecForm["param_" + str(i)].values[:, None]), 1)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"In this sections we will extract the summary metrics AUC, Emax and IC50 from the Sigmoid4_parameters functions"
"These metrics are used as the references to compute the error metrics"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn import metrics
from importlib import reload
import Utils_SummaryMetrics_KLRelevance
import Utils_SummaryMetrics_KLRelevance as MyUtils
reload(Utils_SummaryMetrics_KLRelevance)

"Be careful that x starts from 0.111111 for 9 or 5 drug concentrations in GDSC1 dataset"
"but x starts from 0.142857143 for the case of 7 drug concentrations in GDSC2 dataset"
"The function Get_IC50_AUC_Emax is implemented in Utils_SummaryMetrics_KLRelevance.py to extract the summary metrics"
x_lin = np.linspace(0.111111, 1, 1000)
x_real_dose = np.linspace(0.111111, 1, Dnorm_cell)  #Here is Dnorm_cell due to using GDSC1 that has 9 or 5 doses
Ydose50,Ydose_res,IC50,AUC,Emax = MyUtils.Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose)

# def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
#     plt.figure(fig_num)
#     plt.plot(x_lin, Ydose_res[posy])
#     plt.plot(x_real_dose, y_train_drug[posy, :], '.')
#     plt.plot(IC50[posy], Ydose50[posy], 'rx')
#     plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
#     plt.title(f"AUC = {AUC[posy]}")
#     print(AUC[posy])
#
# posy = 1
# my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Organize summary metrics with proper dimensionality"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUC = np.array(AUC)
IC50 = np.array(IC50)
Emax = np.array(Emax)

"Below we select just the columns with std higher than zero"
Name_Features_Melanoma = df_train_No_MolecForm.columns[start_pos_features:]
Xall = X_train_features.copy()
Yall = y_train_drug.copy()

print("Sanity check to the same features Evelyin Provided")
print(df_ANOVA_feat_Names["feature"].values[start_pos_features:]==Name_Features_Melanoma)

AUC_all = AUC[:, None].copy()
IC50_all = IC50[:, None].copy()
Emax_all = Emax[:, None].copy()

print("AUC train size:", AUC_all.shape)
print("IC50 train size:", IC50_all.shape)
print("Emax train size:", Emax_all.shape)
print("X all data size:", Xall.shape)
print("Y all data size:", Yall.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.model_selection import KFold, cross_val_score
Ndata = Xall.shape[0]
Xind = np.arange(Ndata)
nsplits = 20 #Ndata
k_fold = KFold(n_splits=nsplits)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NegMLL_AllFolds = []
Emax_abs_AllFolds = []
AUC_abs_AllFolds = []
IC50_MSE_AllFolds = []
Med_MSE_AllFolds = []
AllPred_MSE_AllFolds = []
Mean_MSE_AllFolds = []
All_Models = []
Ntasks = Dnorm_cell
list_folds = list(k_fold.split(Xall))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"The for loop below runs the cross-validation process for the MOGP model"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for Nfold in range(0,nsplits+1):
    model = []
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux,(Ntasks,1)), Ylabel_val), 1)

        Xtrain_aux = Xall[train_ind].copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:,None]
        Ytrain = Yall[train_ind].T.flatten().copy()[:,None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()

    else:
        print(f"Train ovell all Data")
        _, test_ind = list_folds[0] #Just assigning by defaul fold0 as the test (of course not to report it as a result)
        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        Xtrain_aux = Xall.copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:,None]
        Ytrain = Yall.T.flatten().copy()[:,None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import GPy
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    mean_all = np.zeros_like(Yval)
    models_outs = []

    rank = int(config.rank)  # Rank for the MultitaskKernel
    "Below we substract one due to being the label associated to the output"
    Dim = Xtrain.shape[1]-1
    myseed = int(config.which_seed)
    np.random.seed(myseed)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Product Kernels. Below we use the locations:"
    "Kernel1 accounts for locations 0:11 for Mutation"
    "Kernel2 accounts for locations 11:end for PANCAN"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    split_dim = 2
    AddKern_loc = [11, Dim]
    mykern = GPy.kern.RBF(AddKern_loc[0], active_dims=list(np.arange(0, AddKern_loc[0])))
    print(list(np.arange(0, AddKern_loc[0])))
    for i in range(1, split_dim):
        mykern = mykern * GPy.kern.RBF(AddKern_loc[i]-AddKern_loc[i-1],active_dims=list(np.arange(AddKern_loc[i-1], AddKern_loc[i])))
        print(list(np.arange(AddKern_loc[i-1], AddKern_loc[i])))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Random Initialization of the kernel hyper-parameters"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    mykern.rbf.lengthscale = float(config.scale)* np.sqrt(Dim) * np.random.rand()
    mykern.rbf.variance.fix()
    for i in range(1,split_dim):
        eval("mykern.rbf_"+str(i)+".lengthscale.setfield(float(config.scale)* np.sqrt(Dim) * np.random.rand(), np.float64)")
        eval("mykern.rbf_" + str(i) + ".variance.fix()")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Create the coregionalization kernel for the MOGP model and create the MOGPRegression model"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    kern = mykern ** GPy.kern.Coregionalize(1, output_dim=Ntasks,rank=rank)
    model = GPy.models.GPRegression(Xtrain, Ytrain, kern)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Random Initialization of the linear combination coefficients for coregionalization matrix of MOGP"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Init_Ws = float(config.weight) * np.random.randn(Ntasks,rank)
    model.kern.coregion.W = Init_Ws
    #model.optimize(optimizer='lbfgsb',messages=True,max_iters=int(config.N_iter))
    #model.optimize()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Here we load the model bash*:
    m_trained = str(config.bash)
    print("loading model ", m_trained)
    #model[:] = np.load('/home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_ANOVA/Best_Model_Drug'+config.drug_name+'_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern/m_' + m_trained + '.npy')
    model[:] = np.load('./Best_Model_Drug' + config.drug_name + '_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern/m_' + m_trained + '.npy')
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    m_pred, v_pred = model.predict(Xval, full_cov=False)
    Yval_curve = Yval.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    m_pred_curve = m_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    v_pred_curve = v_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Here we extract the summary metrics AUC, Emax and IC50 from the MOGP curve predictions"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    x_dose = np.linspace(0.111111, 1.0, Ntasks)
    x_dose_new = np.linspace(0.111111, 1.0, 1000)  #This new lispace is for the pchip interpolation
    Y_pred_interp_all,std_upper_interp_all,std_lower_interp_all,_, Ydose50_pred, IC50_pred, AUC_pred, Emax_pred = MyUtils.Predict_Curve_and_SummaryMetrics(x_dose =x_dose,x_dose_new = x_dose_new,m_pred_curve=m_pred_curve, v_pred=v_pred)

    Ydose50_pred = np.array(Ydose50_pred)
    IC50_pred = np.array(IC50_pred)[:,None]
    AUC_pred = np.array(AUC_pred)[:, None]
    Emax_pred = np.array(Emax_pred)[:, None]
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Compute different error metrics!"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Val_NMLL = -np.mean(model.log_predictive_density(Xval, Yval))  #Negative Log Predictive Density (NLPD)
    Emax_abs = np.mean(np.abs(Emax_val - Emax_pred))
    AUC_abs = np.mean(np.abs(AUC_val - AUC_pred))
    IC50_MSE = np.mean((IC50_val - IC50_pred) ** 2)
    MSE_curves = np.mean((m_pred_curve - Yval_curve) ** 2, 1)
    AllPred_MSE = np.mean((m_pred_curve - Yval_curve) ** 2)

    Med_MSE = np.median(MSE_curves)
    Mean_MSE = np.mean(MSE_curves)

    if Nfold < nsplits:
        print("\nNegLPD Val", Val_NMLL)
        print("IC50 MSE:", IC50_MSE)
        print("AUC MAE:", AUC_abs)
        print("Emax MAE:", Emax_abs)
        print("Med_MSE:", Med_MSE)
        print("Mean_MSE:", Mean_MSE)
        print("All Predictions MSE:", AllPred_MSE)

        NegMLL_AllFolds.append(Val_NMLL.copy())
        Emax_abs_AllFolds.append(Emax_abs.copy())
        AUC_abs_AllFolds.append(AUC_abs.copy())
        IC50_MSE_AllFolds.append(IC50_MSE.copy())
        Med_MSE_AllFolds.append(Med_MSE.copy())
        Mean_MSE_AllFolds.append(Mean_MSE.copy())
        AllPred_MSE_AllFolds.append(AllPred_MSE.copy())

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we prepare the data for Test over Drugs from GDSC2 with 7 Doses, i.e. 1036"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#_FOLDER_GDSC2 = "/home/ac1jjgg/Dataset_BRAF_NoReplica_ANOVA_Features/GDSC2/"
#_FOLDER_GDSC2 = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_BRAF_NoReplica_ANOVA_Features/GDSC2/"
_FOLDER_GDSC2 = "../Dataset_BRAF_NoReplica_ANOVA_Features/GDSC2/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
name_for_KLrelevance_GDSC2 = 'GDSC2_melanoma_BRAF_noreps_v3.csv'

df_train_No_MolecForm_GDSC2 = pd.read_csv(_FOLDER_GDSC2 + name_for_KLrelevance_GDSC2)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_GDSC2 = df_train_No_MolecForm_GDSC2[(df_train_No_MolecForm_GDSC2["DRUG_ID"]==int(config.drug_name))]
try:
    df_train_GDSC2 = df_train_GDSC2.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 25 the input features start
start_pos_features_GDSC2 = 25
print(df_train_GDSC2.columns[start_pos_features_GDSC2])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we use exactly the same features used for the training on GDSC1 their names are in Name_Features_Melanoma"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train_features_GDSC2_NonScaled = df_train_GDSC2[Name_Features_Melanoma].copy()
"Instead of using (MaxMin sclaer) scaler.transform we just extract the min and max and make the transformation"
"This is to directly use the features used for training GDSC1 Dose9 and Dose5"
scaler_sel_max = scaler.data_max_
scaler_sel_min = scaler.data_min_
Xall_GDSC2 = (X_train_features_GDSC2_NonScaled.values-scaler_sel_min)/(scaler_sel_max-scaler_sel_min)

"Below we select just 7 concentration since GDSC2 only has such a number"
y_train_drug_Dose7 = np.clip(df_train_GDSC2["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug_Dose7.shape)
for i in range(2, 8):
    y_train_drug_Dose7 = np.concatenate(
        (y_train_drug_Dose7, np.clip(df_train_GDSC2["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", y_train_drug_Dose7.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train_Dose7 = df_train_GDSC2["param_" + str(1)].values[:, None]
for i in range(2, 5):
    params_4_sig_train_Dose7 = np.concatenate(
        (params_4_sig_train_Dose7, df_train_GDSC2["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics

#plt.close('all')
Kprop = (20.5-3.95)*(1-0.111111)/20.5
xini_adjusted = 0.111111 + ((1-0.111111)-Kprop)
ToMul = (1-0.111111)/Kprop
weighted = (1-0.111111)*ToMul
difference = weighted- (1-0.111111)
x_lin_Dose7 = np.linspace(0.111111, 1, 1000)
x_lin_Dose7_Adjust = np.linspace(xini_adjusted, 1, 1000)
x_real_dose_Dose7 = np.linspace(0.111111-difference, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
x_lin_tile_Dose7 = np.tile(x_lin_Dose7, (params_4_sig_train_Dose7.shape[0], 1))
x_lin_tile_Dose7_Adjust = np.tile(x_lin_Dose7_Adjust, (params_4_sig_train_Dose7.shape[0], 1))
# (x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res_Dose7 = []
AUC_Dose7 = []
IC50_Dose7 = []
Ydose50_Dose7 = []
Emax_Dose7 = []
for i in range(params_4_sig_train_Dose7.shape[0]):
    Ydose_res_Dose7.append(MyUtils.sigmoid_4_param(x_lin_tile_Dose7_Adjust[i, :], *params_4_sig_train_Dose7[i, :]))
    AUC_Dose7.append(metrics.auc(x_lin_tile_Dose7[i, :], Ydose_res_Dose7[i]))
    Emax_Dose7.append(Ydose_res_Dose7[i][-1])
    res1 = (Ydose_res_Dose7[i] < 0.507)
    res2 = (Ydose_res_Dose7[i] > 0.493)
    if (res1 & res2).sum() > 0:
        Ydose50_Dose7.append(Ydose_res_Dose7[i][res1 & res2].mean())
        IC50_Dose7.append(x_lin_Dose7[res1 & res2].mean())
    elif Ydose_res_Dose7[i][-1]<0.5:
       Ydose50_Dose7.append(Ydose_res_Dose7[i].max())
       aux_IC50_Dose7 = x_lin_Dose7[np.where(Ydose_res_Dose7[i].max())[0]][0]  #it has to be a float not an array to avoid bug
       IC50_Dose7.append(aux_IC50_Dose7)
    else:
        Ydose50_Dose7.append(0.5)
        IC50_Dose7.append(1.5) #IC50.append(x_lin[-1])

posy = 0
plt.figure(10)
plt.plot(x_lin_Dose7, Ydose_res_Dose7[posy])
plt.plot(x_real_dose_Dose7, y_train_drug_Dose7[posy, :], '.')
plt.plot(IC50_Dose7[posy], Ydose50_Dose7[posy], 'rx')
plt.plot(x_lin_Dose7, np.ones_like(x_lin_Dose7)*Emax_Dose7[posy], 'r') #Plot a horizontal line as Emax
plt.title(f"AUC Dose7 = {AUC_Dose7[posy]}")
print(AUC_Dose7[posy])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUC_Dose7 = np.array(AUC_Dose7)[:,None]
IC50_Dose7 = np.array(IC50_Dose7)[:,None]
Emax_Dose7 = np.array(Emax_Dose7)[:,None]
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"The Xtest_aux_Dose7 data should have labels of Ntask equal to the size of GDSC1, for instance if using Drug1036"
"then we should assign labels for 9 concentrations, that is because the models trained on GDSC1 presented"
"9 concentration, so when we predict using the model we will obtain 9 predictions."

Xtest_aux_Dose7 = Xall_GDSC2.copy()
Ylabel_test_Dose7 = np.array([i * np.ones(Xtest_aux_Dose7.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
Xtest_GDSC2_Dose7 = np.concatenate((np.tile(Xtest_aux_Dose7, (Ntasks, 1)), Ylabel_test_Dose7), 1)
Ytest_Dose7 = y_train_drug_Dose7.T.flatten().copy()[:, None]

m_pred_Dose7, v_pred_Dose7 = model.predict(Xtest_GDSC2_Dose7, full_cov=False)

Nval_Dose7 = Xtest_aux_Dose7.shape[0]

m_pred_curve_Dose7 = m_pred_Dose7.reshape(Ntasks, Nval_Dose7).T.copy()
v_pred_curve_Dose7 = v_pred_Dose7.reshape(Ntasks, Nval_Dose7).T.copy()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Interpolation of predictions for GDSC2 for Dose7"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if Ntasks == 9:
    print("Remember This is for drugs in GDSC1 with 9 Concentrations")
    assert Ntasks == 9

x_dose_Dose7 = np.linspace(0.111111, 1.0, Ntasks)  # Here 9 due to having 9 dose responses for GDSC1 prediction
x_dose_new_Dose7 = np.linspace(0.111111, 1.0, 1000)
Y_pred_interp_all_Dose7,std_upper_interp_all_Dose7,std_lower_interp_all_Dose7,_, Ydose50_pred_Dose7, IC50_pred_Dose7, AUC_pred_Dose7, Emax_pred_Dose7 = MyUtils.Predict_Curve_and_SummaryMetrics(x_dose =x_dose_Dose7,x_dose_new = x_dose_new_Dose7,m_pred_curve=m_pred_curve_Dose7, v_pred=v_pred_Dose7)

Ydose50_pred_Dose7 = np.array(Ydose50_pred_Dose7)
IC50_pred_Dose7 = np.array(IC50_pred_Dose7)[:, None]
AUC_pred_Dose7 = np.array(AUC_pred_Dose7)[:, None]
Emax_pred_Dose7 = np.array(Emax_pred_Dose7)[:, None]

#posy = 11
plt.figure(60)
plt.plot(x_dose_new_Dose7, Y_pred_interp_all_Dose7[posy])
plt.plot(x_dose_new_Dose7, std_upper_interp_all_Dose7[posy], 'b--')
plt.plot(x_dose_new_Dose7, std_lower_interp_all_Dose7[posy], 'b--')
#plt.plot(x_dose_Dose7, Yval_curve_Dose7[posy, :], '.')
plt.plot(x_real_dose_Dose7, y_train_drug_Dose7[posy, :], '.')
plt.plot(IC50_pred_Dose7[posy], Ydose50_pred_Dose7[posy], 'rx')
plt.plot(x_dose_new_Dose7, np.ones_like(x_dose_new_Dose7) * Emax_pred_Dose7[posy],'b--')  # Plot a horizontal line as Emax
plt.plot(x_dose_new_Dose7, np.ones_like(x_dose_new_Dose7) * Emax_Dose7[posy],'r--')  # Plot a horizontal line as Emax
plt.title(f"AUC Dose7 = {AUC_pred_Dose7[posy]}")
print(AUC_pred_Dose7[posy])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Emax_abs_Dose7 = np.mean(np.abs(Emax_Dose7 - Emax_pred_Dose7))
AUC_abs_Dose7 = np.mean(np.abs(AUC_Dose7 - AUC_pred_Dose7))
IC50_MSE_Dose7 = np.mean((IC50_Dose7 - IC50_pred_Dose7) ** 2)
#MSE_curves_Dose7 = np.mean((m_pred_curve_Dose7 - Yval_curve_Dose7) ** 2, 1)
#AllPred_MSE_Dose7 = np.mean((m_pred_curve_Dose7 - Yval_curve_Dose7) ** 2)
print("IC50 MSE:", IC50_MSE_Dose7)
print("AUC MAE:", AUC_abs_Dose7)
print("Emax MAE:", Emax_abs_Dose7)

from scipy.stats import spearmanr

pos_Actual_IC50 = IC50_Dose7 != 1.5
spear_corr_all, p_value_all = spearmanr(IC50_Dose7, IC50_pred_Dose7)
spear_corr_actualIC50, p_value_actual = spearmanr(IC50_Dose7[pos_Actual_IC50],IC50_pred_Dose7[pos_Actual_IC50])
print("Spearman_all Corr: ", spear_corr_all)
print("Spearman p-value: ", p_value_all)
print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
print("Spearman p-value: ", p_value_actual)