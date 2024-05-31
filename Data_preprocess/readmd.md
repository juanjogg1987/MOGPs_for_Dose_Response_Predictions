# Data Preprocessing Pipeline

This directory contains scripts and notebooks for preprocessing data related to drug response predictions. Please download the raw data into this directory before running the scripts.

## Files and Descriptions

1. **1_GDSC_preprocessing_gdsclC50.R**
   - This R script preprocesses the GDSC (Genomics of Drug Sensitivity in Cancer) dataset to generate initial drug response data.

2. **2_DataPreparation.ipynb**
   - This Jupyter notebook further prepares and cleans the data for subsequent analysis.

3. **3_Fitting_drug_curves.ipynb**
   - This notebook fits drug response curves to the cleaned data, providing essential parameters for analysis.

4. **4_Merging_drug_profiles_with_cell_lines_properties.ipynb**
   - This notebook merges drug response profiles with cell line properties to create a comprehensive dataset.

5. **5_Drug_features_PubChem.ipynb**
   - This notebook retrieves drug features from the PubChem database and integrates them into the dataset.

6. **6_Merge_fitted_data_with_drug_properties.ipynb**
   - This final notebook merges the fitted drug response data with drug properties for the complete analysis dataset.

## Instructions

1. Download and place the raw data files in this directory.
2. Run the scripts and notebooks in the order listed above to preprocess and prepare your data for analysis.

Ensure all required dependencies are installed before running the scripts and notebooks.

---

Feel free to contact Evelyn_Lau@sics.a-star.edu.sg if you have any questions or need further assistance.