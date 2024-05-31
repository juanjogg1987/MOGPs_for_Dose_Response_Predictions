#load libraries
library(gdscIC50)
library(dplyr)

#read GDSC dataset
gdsc1_raw_data <- read.csv("GDSC1_public_raw_data_24Jul22.csv")
gdsc2_raw_data <- read.csv("GDSC2_public_raw_data_24Jul22.csv")

#filter for failed drugs
gdsc1_raw_filtered_1 <- removeFailedDrugs(gdsc1_raw_data)
gdsc2_raw_filtered_1 <- removeFailedDrugs(gdsc2_raw_data)

#filter for missing drugs
gdsc1_raw_filtered_2 <- removeMissingDrugs(gdsc1_raw_filtered_1)
gdsc2_raw_filtered_2 <- removeMissingDrugs(gdsc2_raw_filtered_1)

#Data normalization
#converting the raw fluorescence intensities for the treated wells (the read-out from the assay) to a cell viability value between 0 and 1

normalised_gdsc1raw_filtered <- normalizeData(gdsc1_raw_filtered_2,
                                              trim = T,
                                              neg_control = "NC-0",
                                              pos_control = "B") #negative control is NC-0 for GDSC1

normalised_gdsc2raw_filtered <- normalizeData(gdsc2_raw_filtered_2,
                                              trim = T,
                                              neg_control = "NC-1",
                                              pos_control = "B") #negative control is NC-0 for GDSC2


# add a column of SCAN_DRUG_COSMICID 
normalised_gdsc1raw_filtered <- normalised_gdsc1raw_filtered %>%
  mutate(SCAN_DRUG_COSMICID  = paste(SCAN_ID, DRUG_ID_lib, COSMIC_ID, sep = "_"))

normalised_gdsc2raw_filtered <- normalised_gdsc2raw_filtered %>%
  mutate(SCAN_DRUG_COSMICID  = paste(SCAN_ID, DRUG_ID_lib, COSMIC_ID, sep = "_"))

# Drug treatment concentration scale normalization

################# GDSC1 ################# 

#for GDSC1 with 9 concentration points 2-fold dilution/5-concentration points 4-fold dilution
# to retrieve maxc and x-scale from 1-9 (x = (log(CONC / maxc) / log(2))
scaled_normalised_gdsc1 <- setConcsForNlme(normalised_gdsc1raw_filtered, group_conc_ranges = F)

# Define the function to categorize the dilution range
categorise_df <- function(dilution) {
  # Define the patterns to match
  df_2_fold <- c("D6", "D7", "D8", "D9")
  
  categorise_group <- function(df) {
    
    # Check if any value in normalized_concentrations matches the 2-fold dilution (9-concentration points)
    if (any(df %in% df_2_fold)) {
      return("2")
    } 
    # If neither pattern matches, return as 4-fold dilution
    else {
      return("4")
    }
  }
  # Apply categorize_group to each row of concentrations
  result <- sapply(dilution, categorise_group)
  return(result)
}

# Apply the function to the DataFrame
scaled_normalised_gdsc1 <- scaled_normalised_gdsc1 %>% 
  group_by(SCAN_DRUG_COSMICID) %>% 
  mutate(DF = categorise_df(list(dose))) %>%
  ungroup()

# normalised concentrations scale 0-1 - fd_num
scaled_normalised_gdsc1$fd_num <- scaled_normalised_gdsc1$x/9

# subset columns
subset_scaled_normalised_gdsc1 <- scaled_normalised_gdsc1 %>%
  select(COSMIC_ID, DRUG_ID_lib, CONC, dose, normalized_intensity, SCAN_DRUG_COSMICID, maxc, DF, fd_num)


################# GDSC2 ################# 
# to first retrieve maxc 
scaled_normalised_gdsc2 <- setConcsForNlme(normalised_gdsc2raw_filtered, group_conc_ranges = F)
# remove column x 
scaled_normalised_gdsc2$x <- NULL

### group by 1000-fold range or 1024 fold-range
# first normalize concentrations to a max of 7 first to identify the dilution patterns
scaled_normalised_gdsc2 <- scaled_normalised_gdsc2 %>%
  rowwise() %>%
  mutate(CONC_norm = round((CONC / maxc) * 7, 3))

# Define the function to categorize the dilution range
categorise_FR <- function(concentrations) {
  # Define the patterns to match
  half_log_pattern <- c(0.007, 0.022, 0.070, 0.221, 0.700, 2.214, 7)
  mixed_pattern <- c(0.007, 0.027, 0.109, 0.437, 0.438, 1.75, 3.5,7)
   
  categorise_group <- function(normalised_concentrations) {
    
    # Check if any value in normalized_concentrations matches the half-log pattern within tolerance
    if (all(normalised_concentrations %in% half_log_pattern)) {
      return("1000")
    } 
    # Check if any value in normalized_concentrations matches the mixed pattern within tolerance
    else if (all(normalised_concentrations %in% mixed_pattern)) {
      return("1024")
    } 
    # If neither pattern matches, return "Unknown"
    else {
      return("Unknown")
    }
  }
  # Apply categorize_group to each row of concentrations
  result <- sapply(concentrations, categorise_group)
  return(result)
}
  
# Apply the function to the DataFrame
scaled_normalised_gdsc2  <- scaled_normalised_gdsc2 %>% 
  group_by(SCAN_DRUG_COSMICID) %>% 
  mutate(FOLD_RANGE = categorise_FR(list(CONC_norm))) %>%
  ungroup()

# remove column CONC_norm used for categorising fold-range
scaled_normalised_gdsc2$CONC_norm <- NULL

#for GDSC2 with 7 concentration points (1000-fold-range with half-log dilution)
#for GDSC2 with 7 concentration points (1024-fold-range with  2 x 2-fold dilutions followed by 4 x 4-fold dilutions dilution)
# set as (4x4-fold dilutions, followed by 4x4-fold dilutions) - 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0
scaled_normalised_gdsc2 <- scaled_normalised_gdsc2 %>%
  mutate(x = case_when(
    FOLD_RANGE == "1000" ~ round((log(CONC / maxc) / log(3.16)) + 7, 2), # set from 1-7
    FOLD_RANGE == "1024" ~ round((log(CONC / maxc) / log(4)) + 6, 2) # set as (4x4-fold dilutions, followed by 4x4-fold dilutions) - 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0
  ))

# normalised concentrations scale 0-1 - fd_num

scaled_normalised_gdsc2 <- scaled_normalised_gdsc2 %>%
  mutate(fd_num = case_when(
    FOLD_RANGE == "1000" ~ (x/7), # set from 1-7 in x
    FOLD_RANGE == "1024" ~ (x/6) # set as 4x4-fold dilutions, followed by 4x4-fold dilutions) in x- 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0))
    
    )) 

# subset columns
subset_scaled_normalised_gdsc2 <- scaled_normalised_gdsc2 %>%
  select(COSMIC_ID, DRUG_ID_lib, CONC, dose, normalized_intensity, SCAN_DRUG_COSMICID, maxc, FOLD_RANGE, fd_num)

