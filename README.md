# MOGPs_for_Dose_Response_Predictions

This repository contains an implementation of a Multi-output Gaussian processes (MOGPs) model to predict dose response curves and an implementation of a features relevance determination method based on the Kullback-Leibler divergence. 

The main experiments are presented in: 

- **MOGPTraining_and_KLRelevance_Melanoma_GDSC1_and_GDSC2.ipynb**: This notebook...

- **ExactMOGP_TrainOn_MelGDSC1_ToPredict_MelGDSC2_ANOVAFeatures.ipynb**: This notebook... 

![Overview](figs/Overview_features_cropped.pdf) 

Two datasets, GDSC1 and GDSC2, were constructed by consolidating dose-response data for three drugs (Dabrafenib, PLX-4720, SB590885) targeting the ERK/MAPK pathway, on 277 human cancer cell lines sourced from the GDSC database. Both datasets consist of cancer cell lines representing five different cancer types (BRCA, COAD, SCLC, LUAD and SKCM). Molecular features characterising these cell lines (genetic variations, copy number alterations, DNA methylation) and the chemical properties of the three drugs (sourced from PubChem) were also included. These comprehensive datasets served as input for the MOGP model for predicting full dose-response curves and estimating the relative importance of these input features based on KL divergence. 

![methods](figs/combined_method_architecture.pdf) 


