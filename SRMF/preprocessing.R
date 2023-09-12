#################################################
# This script takes the train and test data and preprocesses them, producing sets of SRMF input matrices which contains:
# 1. response matrix (either IC50, AUC or Emax)
# 2. similarity matrix for chemical features
# 3. similarity matrix for genomic features
#################################################

library(data.table)
library(tidyverse)
library(dplyr)
library(R.matlab)

# packageVersion("rlang")
# packageVersion("tidyr")
# packageVersion("cli")

#################################################
# Define variables
#################################################

metrics <- c("IC50","AUC","Emax")

cellsim_suffix <- "Cellsim_probe.mat"
drugsim_suffix <- "Drugsim_fig_mt.mat"
resp_suffix <- "resp.mat"

#################################################
# Start code
#################################################

setwd("/Users/melodyparker/Documents/DRP/SRMF/GDSC2_Datasets_From5Cancers_Train1Cancer_Increasing_TrainingData")

# Set the directory path
dir_path <- "/Users/melodyparker/Documents/DRP/SRMF/GDSC2_Datasets_From5Cancers_Train1Cancer_Increasing_TrainingData"

# Define outdir and construct its full path
out_dir <- "../SRMF_datasets"
out_dir <- file.path(getwd(), out_dir)
print(out_dir)

# Create "SRMF_datasets" output dir
if (!file.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  print("Directory created!")
} else {
  print("Directory already exists!")
}

# Extract all dataset directories to loop through
datasets <- unlist(strsplit(dir(dir_path, recursive = TRUE), ";"))
datasets <- datasets[grepl("seed", datasets)]

# Loop through IC50, AUC and Emax
for(metric in metrics){
  
  # Loop through datasets
  for (path in datasets) {
    
    # Retrieve variables from the path name
    fields <- strsplit(path, "/")[[1]]
    cancer_type <- fields[1]
    m <- fields[2]
    seed <- fields[3]
    
    # Remove underscore from cancer type string
    cancer_cropped <- gsub("_","",cancer_type)
    
    # Define output sub-directory
    out_sub_dir <- file.path(out_dir,metric,cancer_type,m,seed)
    print(out_sub_dir)
    
    # Create output sub dir
    if (!file.exists(out_sub_dir)) {
      dir.create(out_sub_dir, recursive = TRUE)
      print("Directory created!")
    } else {
      print("Directory already exists!")
    }
    
    # Define test and train input file names
    in_test_file <- paste0(cancer_type,"/Test_",cancer_cropped,".csv")
    in_train_file <- paste0(cancer_type,"/",m,"/",seed,"/",m,"_",seed,".csv")
    
    # Read in test and train datasets
    test_dataset <- fread(in_test_file)
    print(paste0("Reading in ", in_test_file))
    train_dataset <- fread(in_train_file)
    print(paste0("Reading in ", in_train_file))
    
    #################################################
    # 1. Create response matrix for merged train and test data
    #################################################
    
    # Define out file path for the merged response matrices
    out_merged_train_mat <- file.path(out_sub_dir,paste(cancer_type,m,seed,metric,resp_suffix,sep="_"))
    out_merged_train_file <- file.path(out_sub_dir,paste(cancer_type,m,seed,metric,"resp.csv",sep="_"))
    out_merged_test_file <- file.path(out_sub_dir,paste("Test",cancer_type,m,seed,metric,"resp.csv",sep="_"))
    
    # Clone the test dataset then change the metric column to NaN
    test_dataset_NaN <- test_dataset
    test_dataset_NaN[,metric] <- NaN
    
    # Bind the train and NaN-test datasets
    merged_train_dataset <- rbind(train_dataset, test_dataset_NaN)
    
    # Clone the train dataset then change the metric column to NaN
    # Make sure the rows and cols are the same as for the other merged dataset
    train_dataset_NaN <- train_dataset
    train_dataset_NaN[,metric] <- NaN
    
    # Bind the NaN-train and test datasets
    merged_test_dataset <- rbind(train_dataset_NaN, test_dataset)
    
    # Transform from long to wide for train data
    merged_train_subset <- merged_train_dataset[,c('DRUG_ID','COSMIC_ID',..metric)]
    wide_train <- merged_train_subset %>% pivot_wider(names_from = DRUG_ID, values_from = metric)
    wide_train <- as.data.frame(wide_train)
    rownames(wide_train) <- wide_train[,"COSMIC_ID"]; wide_train <- wide_train[,-1]
    
    # Change NA values to NaN (SRMF requires NaNs)
    wide_train[is.na(wide_train)] <- NaN
    
    # Create a merged response matrix for test data
    merged_test_subset <- merged_test_dataset[,c('DRUG_ID','COSMIC_ID',..metric)]
    wide_test <- merged_test_subset %>% pivot_wider(names_from = DRUG_ID, values_from = metric)
    wide_test <- as.data.frame(wide_test)
    rownames(wide_test) <- wide_test[,"COSMIC_ID"]; wide_test <- wide_test[,-1]
    
    # Write test and train to csv file with column and row names
    write.csv(wide_train,out_merged_train_file)
    write.csv(wide_test,out_merged_test_file)
    
    # Convert to matrix
    merged_mat <- as.matrix(wide_train)
    
    # Write to matlab mat file
    writeMat(out_merged_train_mat,resp=merged_mat)
    
    #################################################
    # 2. Create the merged drug similarity matrix
    #################################################
    
    # Define drugs output file
    out_drugsim_file <- file.path(out_sub_dir,paste(cancer_type,m,seed,metric,drugsim_suffix,sep="_"))
    
    # Minus 1 because we will get rid of drug names
    a <- which(colnames(merged_train_dataset) == "2bonds") -1
    b <- which(colnames(merged_train_dataset) == "heavy_atom_count") -1
    
    # Remove duplicates
    chem_subset <- merged_train_dataset[!duplicated(merged_train_dataset$DRUG_ID), ]
    chem_subset <- as.data.frame(chem_subset)
    
    # Subset to only the chemical data with drug id as row names
    chem <- chem_subset %>%
      column_to_rownames('DRUG_ID') %>%
      select(c(a:b)) 
    
    # Create correlation matrix
    chem_cov_mat <- cor(t(chem))
    
    # Write drugs correlation matrix to matlab file
    writeMat(out_drugsim_file,Drugsim_fig_mt=chem_cov_mat)
    
    #################################################
    # 3. Create similarity matrix for the genomic features
    #################################################
    
    # Out file name for cell line features similarity matrix
    out_cellsim_file <- file.path(out_sub_dir,paste(cancer_type,m,seed,metric,cellsim_suffix,sep="_"))
    
    # Pick the range for the genomic feature data
    # ABCB1_mut is the first gen feat col, chr8:95652455-95652873(ESRP1)_HypMET is the last
    # Minus 1 because we will get rid of cosmic id column
    c <- which(colnames(merged_train_dataset) == "ABCB1_mut") -1
    d <- which(colnames(merged_train_dataset) == "chr8:95652455-95652873(ESRP1)_HypMET") -1
    
    # Remove duplicate genomic feature data data
    gen_subset <- merged_train_dataset[!duplicated(merged_train_dataset$COSMIC_ID), ]
    gen_subset <- as.data.frame(gen_subset)
    
    # Subset to only the genomic feature data with cosmic id as row names
    gen <- gen_subset %>%
      column_to_rownames('COSMIC_ID') %>%
      select(c(c:d)) 
    
    # Create Pearson correlation coefficient matrix for genetic features
    gen_cov_mat <- cor(t(gen))
    
    # Write cell line correlation matrix to matlab matrix file
    writeMat(out_cellsim_file,Cellsim_probe=gen_cov_mat)
    
  }
}
