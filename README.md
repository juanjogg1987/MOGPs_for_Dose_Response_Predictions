# MOGPs_for_Dose_Response_Predictions

This repository contains an implementation of a Multi-output Gaussian processes (MOGPs) model to predict dose response curves and an implementation of a features relevance determination method based on the Kullback-Leibler divergence. 

The main experiments are presented in: 

- **MOGPTraining_and_KLRelevance_Melanoma_GDSC1_and_GDSC2.ipynb**: This notebook...

- **ExactMOGP_TrainOn_MelGDSC1_ToPredict_MelGDSC2_ANOVAFeatures.ipynb**: This notebook... 

![Overview](figs/Overview_features.jpg) 

Two datasets, GDSC1 and GDSC2, were constructed by consolidating dose-response data for three drugs (Dabrafenib, PLX-4720, SB590885) targeting the ERK/MAPK pathway, on 277 human cancer cell lines sourced from the GDSC database. Both datasets consist of cancer cell lines representing five different cancer types (BRCA, COAD, SCLC, LUAD and SKCM). Molecular features characterising these cell lines (genetic variations, copy number alterations, DNA methylation) and the chemical properties of the three drugs (sourced from PubChem) were also included. These comprehensive datasets served as input for the MOGP model for predicting full dose-response curves and estimating the relative importance of these input features based on KL divergence. 

![methods](figs/combined_method_architecture.jpg) 

(a) Kullback-Leibler relevance determination to estimate feature importance. To compute the relevance of a feature w.r.t a data observation we have to make two predictions, one for the original observation x, and another where such an observation is subtly modified by a small Δ on the p-th feature, xΔp. The MOGP outputs two distributions, one for input x and another for xΔp, then a the DKL[.||.] module computes a divergence between both predictive distributions and then a normalisation applies the operation 2DKL[.||.]/Δ to obtain the relevance of the p-th feature (see section Kullback-Leibler Relevance Determination for additional details). (b) Prediction of full dose-response curves using MOGP. The input vector x* is composed of: the cell line genomic features, mutation, methylation and copy number, and the drug compounds. The vector x* feeds two blocks of the MOGP prediction, such blocks generate the mean (x*) (red-ish vector) and covariance S(x*) (red-ish matrix); both blocks are a pictorial representation of equations (3) and (4). The last panel to the right hand side shows the prediction of a melanoma cancer cell line with Comic-ID: 1240226 from the GDSC2 dataset and treated with PLX-4720. The mean vector (x*) has a size of D=7, each of its entries represent the cell viability of the d-th drug concentration (black dots). The covariance matrix S(x*) encodes the uncertainty of the prediction; it can (loosely) be expressed as the dashed red line that accounts for a confidence interval of two standard deviations. The multiple coloured functions amongst the dashed red line depict random samples taken from the predictive distribution, N(y*|(X*),S(X*)), to exemplify the stochastic nature of the MOGP prediction.
