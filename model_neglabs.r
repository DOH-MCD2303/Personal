# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Machine Learning Model for Record Linkage
# MAGIC 
# MAGIC This code links the newly reported negative lab test results to the existing negative test results and deduplicates the new test results that did not link to existing results.
# MAGIC 
# MAGIC Created by: Matt Doxey
# MAGIC Created on: 03/2021
# MAGIC Updated: 08/2021

# COMMAND ----------

# MAGIC %python
# MAGIC ## setup environment
# MAGIC 
# MAGIC library(SparkR)
# MAGIC sparkR.session()
# MAGIC 
# MAGIC library(readr)
# MAGIC library(RecordLinkage)
# MAGIC library(ipred)
# MAGIC library(data.table)
# MAGIC library(dplyr)
# MAGIC library(plyr)
# MAGIC library(magrittr)
# MAGIC library(readxl)
# MAGIC library(readr)
# MAGIC library(xlsx)
# MAGIC library(sqldf)
# MAGIC library(tidyverse)
# MAGIC library(janitor)
# MAGIC library(haven)
# MAGIC library(lubridate)
# MAGIC library(igraph)
# MAGIC library(caret)
# MAGIC library(caTools)

# COMMAND ----------

# MAGIC %python
# MAGIC ## LOAD old model objects and data - Connect to VPN to access Y drive!
# MAGIC 
# MAGIC load("Y:/Confidential/DCHS/CDE/01_Linelists_Cross Coverage/Novel CoV/Negative Test Data Management/Negative Labs New System/LinkageModels/link_model.RData")
# MAGIC load("Y:/Confidential/DCHS/CDE/01_Linelists_Cross Coverage/Novel CoV/Negative Test Data Management/Negative Labs New System/LinkageModels/names_model.RData")
# MAGIC load("Y:/Confidential/DCHS/CDE/01_Linelists_Cross Coverage/Novel CoV/Negative Test Data Management/Negative Labs New System/LinkageModels/neglabs_model.RData")
# MAGIC load("Y:/Confidential/DCHS/CDE/01_Linelists_Cross Coverage/Novel CoV/Negative Test Data Management/Negative Labs New System/LinkageModels/trainedset_link.RData")
# MAGIC load("Y:/Confidential/DCHS/CDE/01_Linelists_Cross Coverage/Novel CoV/Negative Test Data Management/Negative Labs New System/LinkageModels/trainedset_names.RData")
# MAGIC 
# MAGIC ## extract old model datasets
# MAGIC 
# MAGIC dt_train_old_1 <- trainedset_link$data1
# MAGIC dt_train_old_2 <- trainedset_link$data2
# MAGIC 
# MAGIC dt_train_old_names <- trainedset_names$data
# MAGIC 
# MAGIC str(dt_train_old_1)
# MAGIC 
# MAGIC ## import neglabs data (saved copy from ASE)
# MAGIC 
# MAGIC newlabs_raw <- as.data.table(read.csv("~/Projects/model_ml/newlabs_raw.csv")) ## 
# MAGIC oldlabs_raw <- as.data.table(read.csv("~/Projects/model_ml/oldlabs_sample_raw.csv")) ## ----> sample of old negative labs
# MAGIC 
# MAGIC # rename
# MAGIC 
# MAGIC dt_new <- newlabs_raw
# MAGIC dt_old <- oldlabs_raw
# MAGIC 
# MAGIC 
# MAGIC ## TRANSFORM
# MAGIC 
# MAGIC ## Pre-process
# MAGIC 
# MAGIC ## Get random sample of old data (very large!!)
# MAGIC 
# MAGIC set.seed(1222)
# MAGIC dt_wdrs_s <- sample(dt_wdrs, 100000, replace = TRUE)
# MAGIC 
# MAGIC # Need to fix NA values
# MAGIC dt_wdrs$PATIENT_FIRSTNAME <- as.character(dt_wdrs$PATIENT_FIRSTNAME)
# MAGIC dt_wdrs$PATIENT_FIRSTNAME[is.na(dt_wdrs$PATIENT_FIRSTNAME)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_MIDNAME <- as.character(dt_wdrs$PATIENT_MIDNAME)
# MAGIC dt_wdrs$PATIENT_MIDNAME[is.na(dt_wdrs$PATIENT_MIDNAME)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_LASTNAME <- as.character(dt_wdrs$PATIENT_LASTNAME)
# MAGIC dt_wdrs$PATIENT_LASTNAME[is.na(dt_wdrs$PATIENT_LASTNAME)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADDRESS_1 <- as.character(dt_wdrs$PATIENT_ADDRESS_1)
# MAGIC dt_wdrs$PATIENT_ADDRESS_1[is.na(dt_wdrs$PATIENT_ADDRESS_1)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADDRESS_2 <- as.character(dt_wdrs$PATIENT_ADDRESS_2)
# MAGIC dt_wdrs$PATIENT_ADDRESS_2[is.na(dt_wdrs$PATIENT_ADDRESS_2)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADDRESS_CITY <- as.character(dt_wdrs$PATIENT_ADDRESS_CITY)
# MAGIC dt_wdrs$PATIENT_ADDRESS_CITY[is.na(dt_wdrs$PATIENT_ADDRESS_CITY)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADDRESS_STATE <- as.character(dt_wdrs$PATIENT_ADDRESS_STATE)
# MAGIC dt_wdrs$PATIENT_ADDRESS_STATE[is.na(dt_wdrs$PATIENT_ADDRESS_STATE)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADDRESS_ZIP <- as.character(dt_wdrs$PATIENT_ADDRESS_ZIP)
# MAGIC dt_wdrs$PATIENT_ADDRESS_ZIP[is.na(dt_wdrs$PATIENT_ADDRESS_ZIP)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_PHONE_NUM <- as.character(dt_wdrs$PATIENT_PHONE_NUM)
# MAGIC dt_wdrs$PATIENT_PHONE_NUM[is.na(dt_wdrs$PATIENT_PHONE_NUM)] <- ""
# MAGIC 
# MAGIC dt_wdrs$PATIENT_ADMINISTRATIVE_SEX <- as.character(dt_wdrs$PATIENT_ADMINISTRATIVE_SEX)
# MAGIC dt_wdrs$PATIENT_ADMINISTRATIVE_SEX[is.na(dt_wdrs$PATIENT_ADMINISTRATIVE_SEX)] <- ""
# MAGIC 
# MAGIC dt_wdrs$SSN <- as.character(dt_wdrs$SSN)
# MAGIC dt_wdrs$SSN[is.na(dt_wdrs$SSN)] <- ""
# MAGIC 
# MAGIC dt_wdrs$MEDREC <- as.character(dt_wdrs$MEDREC)
# MAGIC dt_wdrs$MEDREC[is.na(dt_wdrs$MEDREC)] <- ""
# MAGIC 
# MAGIC dt_wdrs$WELRS_ASSIGNED_COUNTY <- as.character(dt_wdrs$WELRS_ASSIGNED_COUNTY)
# MAGIC dt_wdrs$WELRS_ASSIGNED_COUNTY[is.na(dt_wdrs$WELRS_ASSIGNED_COUNTY)] <- ""
# MAGIC 
# MAGIC dt_wdrs$WELRS_OBX_ID <- as.character(dt_wdrs$WELRS_OBX_ID)
# MAGIC dt_wdrs$WELRS_OBX_ID[is.na(dt_wdrs$WELRS_OBX_ID)] <- ""
# MAGIC 
# MAGIC # Need to fix NA values
# MAGIC dt_new$PATIENT_FIRSTNAME <- as.character(dt_new$PATIENT_FIRSTNAME)
# MAGIC dt_new$PATIENT_FIRSTNAME[is.na(dt_new$PATIENT_FIRSTNAME)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_MIDNAME <- as.character(dt_new$PATIENT_MIDNAME)
# MAGIC dt_new$PATIENT_MIDNAME[is.na(dt_new$PATIENT_MIDNAME)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_LASTNAME <- as.character(dt_new$PATIENT_LASTNAME)
# MAGIC dt_new$PATIENT_LASTNAME[is.na(dt_new$PATIENT_LASTNAME)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADDRESS_1 <- as.character(dt_new$PATIENT_ADDRESS_1)
# MAGIC dt_new$PATIENT_ADDRESS_1[is.na(dt_new$PATIENT_ADDRESS_1)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADDRESS_2 <- as.character(dt_new$PATIENT_ADDRESS_2)
# MAGIC dt_new$PATIENT_ADDRESS_2[is.na(dt_new$PATIENT_ADDRESS_2)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADDRESS_CITY <- as.character(dt_new$PATIENT_ADDRESS_CITY)
# MAGIC dt_new$PATIENT_ADDRESS_CITY[is.na(dt_new$PATIENT_ADDRESS_CITY)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADDRESS_STATE <- as.character(dt_new$PATIENT_ADDRESS_STATE)
# MAGIC dt_new$PATIENT_ADDRESS_STATE[is.na(dt_new$PATIENT_ADDRESS_STATE)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADDRESS_ZIP <- as.character(dt_new$PATIENT_ADDRESS_ZIP)
# MAGIC dt_new$PATIENT_ADDRESS_ZIP[is.na(dt_new$PATIENT_ADDRESS_ZIP)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_PHONE_NUM <- as.character(dt_new$PATIENT_PHONE_NUM)
# MAGIC dt_new$PATIENT_PHONE_NUM[is.na(dt_new$PATIENT_PHONE_NUM)] <- ""
# MAGIC 
# MAGIC dt_new$PATIENT_ADMINISTRATIVE_SEX <- as.character(dt_new$PATIENT_ADMINISTRATIVE_SEX)
# MAGIC dt_new$PATIENT_ADMINISTRATIVE_SEX[is.na(dt_new$PATIENT_ADMINISTRATIVE_SEX)] <- ""
# MAGIC 
# MAGIC dt_new$SSN <- as.character(dt_new$SSN)
# MAGIC dt_new$SSN[is.na(dt_new$SSN)] <- ""
# MAGIC 
# MAGIC dt_new$MEDREC <- as.character(dt_new$MEDREC)
# MAGIC dt_new$MEDREC[is.na(dt_new$MEDREC)] <- ""
# MAGIC 
# MAGIC dt_new$WELRS_ASSIGNED_COUNTY <- as.character(dt_new$WELRS_ASSIGNED_COUNTY)
# MAGIC dt_new$WELRS_ASSIGNED_COUNTY[is.na(dt_new$WELRS_ASSIGNED_COUNTY)] <- ""
# MAGIC 
# MAGIC dt_new$WELRS_OBX_ID <- as.character(dt_new$WELRS_OBX_ID)
# MAGIC dt_new$WELRS_OBX_ID[is.na(dt_new$WELRS_OBX_ID)] <- ""
# MAGIC 
# MAGIC # Added on 1/14 - sorted by MSH_ID
# MAGIC #dt_new <- dt_new[order(dt_new$MSH_ID, dt_new$WELRS_OBX_ID),] ## ----> turns dt_new into a vector; bad!
# MAGIC 
# MAGIC dt_old$PATIENT_FIRSTNAME <- as.character(dt_old$PATIENT_FIRSTNAME)
# MAGIC dt_old$PATIENT_FIRSTNAME[is.na(dt_old$PATIENT_FIRSTNAME)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_MIDNAME <- as.character(dt_old$PATIENT_MIDNAME)
# MAGIC dt_old$PATIENT_MIDNAME[is.na(dt_old$PATIENT_MIDNAME)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_LASTNAME <- as.character(dt_old$PATIENT_LASTNAME)
# MAGIC dt_old$PATIENT_LASTNAME[is.na(dt_old$PATIENT_LASTNAME)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADDRESS_1 <- as.character(dt_old$PATIENT_ADDRESS_1)
# MAGIC dt_old$PATIENT_ADDRESS_1[is.na(dt_old$PATIENT_ADDRESS_1)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADDRESS_2 <- as.character(dt_old$PATIENT_ADDRESS_2)
# MAGIC dt_old$PATIENT_ADDRESS_2[is.na(dt_old$PATIENT_ADDRESS_2)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADDRESS_CITY <- as.character(dt_old$PATIENT_ADDRESS_CITY)
# MAGIC dt_old$PATIENT_ADDRESS_CITY[is.na(dt_old$PATIENT_ADDRESS_CITY)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADDRESS_STATE <- as.character(dt_old$PATIENT_ADDRESS_STATE)
# MAGIC dt_old$PATIENT_ADDRESS_STATE[is.na(dt_old$PATIENT_ADDRESS_STATE)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADDRESS_ZIP <- as.character(dt_old$PATIENT_ADDRESS_ZIP)
# MAGIC dt_old$PATIENT_ADDRESS_ZIP[is.na(dt_old$PATIENT_ADDRESS_ZIP)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_PHONE_NUM <- as.character(dt_old$PATIENT_PHONE_NUM)
# MAGIC dt_old$PATIENT_PHONE_NUM[is.na(dt_old$PATIENT_PHONE_NUM)] <- ""
# MAGIC 
# MAGIC dt_old$PATIENT_ADMINISTRATIVE_SEX <- as.character(dt_old$PATIENT_ADMINISTRATIVE_SEX)
# MAGIC dt_old$PATIENT_ADMINISTRATIVE_SEX[is.na(dt_old$PATIENT_ADMINISTRATIVE_SEX)] <- ""
# MAGIC 
# MAGIC dt_old$SSN <- as.character(dt_old$SSN)
# MAGIC dt_old$SSN[is.na(dt_old$SSN)] <- ""
# MAGIC 
# MAGIC dt_old$MEDREC <- as.character(dt_old$MEDREC)
# MAGIC dt_old$MEDREC[is.na(dt_old$MEDREC)] <- ""
# MAGIC 
# MAGIC dt_old$WELRS_ASSIGNED_COUNTY <- as.character(dt_old$WELRS_ASSIGNED_COUNTY)
# MAGIC dt_old$WELRS_ASSIGNED_COUNTY[is.na(dt_old$WELRS_ASSIGNED_COUNTY)] <- ""
# MAGIC 
# MAGIC dt_old$WELRS_OBX_ID <- as.character(dt_old$WELRS_OBX_ID)
# MAGIC dt_old$WELRS_OBX_ID[is.na(dt_old$WELRS_OBX_ID)] <- ""
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Prepare New Labs to Link
# MAGIC 
# MAGIC newlabs_tolink <- r_newlabs %>%
# MAGIC   mutate(neg_pid = NA_character_) %>%  # Add an empty neg_pid field, so that the fields match oldlabs.
# MAGIC   clean_names() %>%
# MAGIC   select("welrs_obx_id", "msh_id", "neg_pid", "patient_firstname", "patient_midname", "patient_lastname", 
# MAGIC          "patient_phone_num", "patient_date_of_birth", "patient_administrative_sex", "ssn", "medrec",
# MAGIC          "patient_address_1", "patient_address_2", "patient_address_city", "patient_address_state",
# MAGIC          "patient_address_zip", "welrs_assigned_county") %>%
# MAGIC   prep_names() %>%
# MAGIC   prep_phonenumbers() %>%
# MAGIC   prep_SSNs() %>%
# MAGIC   prep_addresses() %>%
# MAGIC   prep_birthdates() %>% 
# MAGIC   prep_sex() %>%
# MAGIC   mutate(medrec = ifelse(medrec == "", NA_character_, medrec)) %>%
# MAGIC   ## Remove records with invalid names:
# MAGIC   filter(!(firstname == "TB" & lastname == "E"), !(firstname == "" | lastname == ""),
# MAGIC          !(is.na(firstname) | is.na(lastname)),
# MAGIC          !(firstname == "EO" & lastname == "EMPLOYEE"),
# MAGIC          !(firstname == "CSL" & lastname == "CSL"),
# MAGIC          !(str_sub(firstname, 1, 2) == "AH" & lastname == "UNOS"),
# MAGIC          !(lastname == "DOE")) %>%
# MAGIC   ## Reorder the fields to make a visual comparison of them easier during manual scoring.
# MAGIC   select(welrs_obx_id, msh_id, neg_pid, dob, firstname, middlename, lastname, sex, phonenumber, ssn9, ssn4
# MAGIC          , medrec, street, city, zipcode, county, state, firstname_sdx, lastname_sdx, subname)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## GET data
# MAGIC 
# MAGIC ## new labs and wdrs
# MAGIC newlabs_clean <- as.data.table(read.csv("~/Projects/model_ml/neglabs_cleaned.csv")) ## ----> new negative lab records
# MAGIC wdrs_clean <- as.data.table(read.csv("~/Projects/model_ml/wdrs_sample_cleaned.csv")) ## ----> new WDRS records
# MAGIC 
# MAGIC dt_neg <- newlabs_clean
# MAGIC dt_wdrs <- wdrs_clean
# MAGIC 
# MAGIC ## Standardize WDRS dataset
# MAGIC 
# MAGIC wdrs_tolink_new <- dt_wdrs %>%
# MAGIC   rename(patient_date_of_birth = BIRTH_DATE, patient_firstname = FIRST_NAME,
# MAGIC          patient_midname = middle_name, patient_lastname = LAST_NAME,
# MAGIC          welrs_assigned_county = WELRS__ASSIGNED__COUNTY,
# MAGIC          row_id = DOH_DD_PK) %>%
# MAGIC   recode_county() %>%
# MAGIC   clean_names() %>%
# MAGIC   select(row_id, patient_firstname, patient_midname, patient_lastname, 
# MAGIC          patient_phone_num, patient_date_of_birth, patient_administrative_sex, ssn, medrec,
# MAGIC          patient_address_1, patient_address_2, patient_address_city, patient_address_state,
# MAGIC          patient_address_zip, welrs_assigned_county)
# MAGIC   
# MAGIC wdrs_tolink_new <- wdrs_tolink_new %>%
# MAGIC   prep_names() %>%
# MAGIC   prep_phonenumbers() %>%
# MAGIC   prep_SSNs() %>%
# MAGIC   prep_addresses() %>%
# MAGIC   #prep_birthdates() %>%
# MAGIC   #filter(!is.na(dob)) %>%
# MAGIC   prep_sex()
# MAGIC 
# MAGIC setnames(wdrs_tolink_new, "patient_date_of_birth", "dob")
# MAGIC   
# MAGIC wdrs_tolink <- wdrs_tolink_new[, c("row_id", "dob", "firstname", "middlename", "lastname", "sex", "phonenumber", "ssn9"
# MAGIC                                        , "ssn4", "medrec", "street", "city", "zipcode", "county", "state", "firstname_sdx" 
# MAGIC                                        , "lastname_sdx", "subname")]
# MAGIC 
# MAGIC dt_wdrs <- wdrs_tolink
# MAGIC 
# MAGIC col_wdrs <- sort(colnames(dt_wdrs))
# MAGIC 
# MAGIC ## sample of old negative labs data
# MAGIC oldlabs_raw <- as.data.table(read.csv("~/Projects/model_ml/oldlabs_sample_raw.csv"))
# MAGIC dt_old <- as.data.table(oldlabs_raw)
# MAGIC 
# MAGIC oldlabs_tolink <- dt_old %>%
# MAGIC   clean_names() %>%
# MAGIC   filter(str_sub(neg_pid, 1, 1) != "N") %>%
# MAGIC   select(welrs_obx_id, msh_id, neg_pid, patient_firstname, patient_midname, patient_lastname, 
# MAGIC          patient_phone_num, patient_date_of_birth, patient_administrative_sex, ssn, medrec,
# MAGIC          patient_address_1, patient_address_2, patient_address_city, patient_address_state,
# MAGIC          patient_address_zip, welrs_assigned_county)
# MAGIC 
# MAGIC oldlabs_tolink <- oldlabs_tolink %>%
# MAGIC   prep_names() %>%
# MAGIC   prep_phonenumbers() %>%
# MAGIC   prep_SSNs() %>%
# MAGIC   prep_addresses() %>%
# MAGIC   #prep_birthdates() %>%
# MAGIC   prep_sex() %>%
# MAGIC   mutate(medrec = ifelse(medrec == "", NA_character_, medrec))
# MAGIC 
# MAGIC setnames(oldlabs_tolink, "patient_date_of_birth", "dob")
# MAGIC 
# MAGIC ## reorder the fields to make a visual comparison of them easier during manual scoring.
# MAGIC oldlabs_tolink <- oldlabs_tolink %>%
# MAGIC   select(welrs_obx_id, msh_id, neg_pid, dob, firstname, middlename, lastname, sex,
# MAGIC          phonenumber, ssn9, ssn4, medrec, street, city, zipcode, county, state, firstname_sdx,
# MAGIC          lastname_sdx, subname)
# MAGIC 
# MAGIC ## create row_id variable
# MAGIC 
# MAGIC oldlabs_tolink <- mutate(oldlabs_tolink, row_id = paste0(welrs_obx_id, "_", msh_id))
# MAGIC 
# MAGIC 
# MAGIC col_old <- sort(colnames(oldlabs_tolink))
# MAGIC 
# MAGIC ## COMPARE to old method input data
# MAGIC 
# MAGIC ## column names
# MAGIC glimpse(dt_neg)
# MAGIC str(dt_neg)
# MAGIC str(dt_train_old_1)
# MAGIC 
# MAGIC col_new <- sort(colnames(dt_neg))
# MAGIC col_old_1 <- sort(colnames(dt_train_old_1))
# MAGIC col_old_2 <- sort(colnames(dt_train_old_2))
# MAGIC 
# MAGIC col_new
# MAGIC col_old_1
# MAGIC col_old_2
# MAGIC 
# MAGIC setdiff(col_new, col_old_1)
# MAGIC setdiff(col_new, col_old_2)
# MAGIC 
# MAGIC ## medrec and other unique identifiers
# MAGIC table(is.na(dt_neg$medrec))
# MAGIC table(dt_neg[, dt_neg == ""])
# MAGIC table(is.na(dt_train_old_1$medrec))
# MAGIC table(is.na(dt_train_old_2$medrec))
# MAGIC 
# MAGIC table(is.na(dt_neg$ssn9))
# MAGIC table(dt_neg[, ssn9 == ""])
# MAGIC table(dt_neg[, ssn9 == "null"])
# MAGIC 
# MAGIC table(is.na(dt_train_old_1$ssn9))
# MAGIC table(is.na(dt_train_old_2$ssn9))
# MAGIC 
# MAGIC 
# MAGIC ## BREAK DOWN birthdate column
# MAGIC 
# MAGIC ## negative labs, new
# MAGIC dt_neg$by <- year(ymd(dt_neg$dob))
# MAGIC dt_neg$bm <- month(ymd(dt_neg$dob))
# MAGIC dt_neg$bd <- day(ymd(dt_neg$dob))
# MAGIC 
# MAGIC ## negative labs, old 
# MAGIC dt_old$by <- year(ymd(dt_old$dob))
# MAGIC dt_old$bm <- month(ymd(dt_old$dob))
# MAGIC dt_old$bd <- day(ymd(dt_old$dob))
# MAGIC 
# MAGIC ## wdrs records 
# MAGIC dt_wdrs$by <- year(ymd(dt_wdrs$dob))
# MAGIC dt_wdrs$bm <- month(ymd(dt_wdrs$dob))
# MAGIC dt_wdrs$bd <- day(ymd(dt_wdrs$dob))
# MAGIC 
# MAGIC 
# MAGIC ## SAVE FILES!!! Intermediate stop point!!! ##
# MAGIC 
# MAGIC write.csv(dt_neg, "~/Projects/model_ml/inputs/neglabs_prepped_new.csv") ## ----> sample of new negative lab files ready for model input
# MAGIC write.csv(dt_old, "~/Projects/model_ml/inputs/neglabs_prepped_old.csv") ## ----> sample of new negative lab files ready for model input
# MAGIC write.csv(dt_wdrs, "~/Projects/model_ml/inputs/wdrs_sample_prepped.csv") ## ----> sample of wdrs records
# MAGIC 
# MAGIC write.csv(dt_train_old_1, "~/Projects/model_ml/inputs/oldmodel_trainingset_1.csv") ## ----> old model training dataset #1
# MAGIC write.csv(dt_train_old_2, "~/Projects/model_ml/inputs/oldmodel_trainingset_2.csv") ## ----> old model training dataset #2
# MAGIC write.csv(dt_train_old_names, "~/Projects/model_ml/inputs/oldmodel_trainingset_names.csv") ## ----> old model training dataset for phonetic names
# MAGIC 
# MAGIC 
# MAGIC ## ---------------------------
# MAGIC ## UPDATED MODEL -------------
# MAGIC ## ---------------------------
# MAGIC 
# MAGIC ## 3/4/2020 - the idea behind this model is to create an updated data set similar to the original probabilistic matching model,
# MAGIC ## so that a comparison can be made between the old model and a new model that ONLY updates and makes slight adjustments to the dataset
# MAGIC ## This will allow us to measure the impact of a new dataset and will act as a starting point for additional model testing and comparison
# MAGIC ## Method: a sample of the negative lab data will be joined with a sample of the WDRS data to create training, testing, and validation datasets.
# MAGIC ## The traiing datasets will be the result of the two datasets (neglabs and wdrs) de-duplicated with weighted pairs included. 
# MAGIC ## The output of thededuplication process will be used as a training dataset for the model and will be compared against both the validation dataset
# MAGIC ## as well as the old training dataset. Once the dataset has been refined, and a new model trained, it will be tested against the original model. 
# MAGIC 
# MAGIC ## Steps are as follows:
# MAGIC ## 1) Create training and testing sets
# MAGIC ## 2) De-duplicate the training set 
# MAGIC ##    a) string comparison
# MAGIC ##    b) phonetic comparison
# MAGIC ## 3) Generate pairs for the updated model
# MAGIC ## 4) Create supervised training set
# MAGIC ## 5) Train model
# MAGIC ## 6) Refine model
# MAGIC ##    a) leave-one out cross validation
# MAGIC ##    b) k-fold cross validation
# MAGIC ## 6) Validate model against validation set (current data?)
# MAGIC ## 6) Compare model results
# MAGIC 
# MAGIC ## ----------------------------------------------
# MAGIC 
# MAGIC 
# MAGIC ## 1) CREATE training, testing, and validation sets
# MAGIC 
# MAGIC 
# MAGIC ## combine neglabs and wdrs datasets
# MAGIC setdiff(col_new, col_wdrs)
# MAGIC 
# MAGIC dt_all <- rbind.fill(dt_neg, dt_wdrs)
# MAGIC 
# MAGIC ## approach #1
# MAGIC #set.seed(1230)
# MAGIC 
# MAGIC #train_index <- createDataPartition(dataset$Species, p = 0.80, list=FALSE)
# MAGIC #validation_set <- dataset[-validation_index,]
# MAGIC #training_set <- dataset[validation_index,]
# MAGIC 
# MAGIC ## approach #2
# MAGIC n <- nrow(dt_all)
# MAGIC n_train <- round(0.70 * n)
# MAGIC 
# MAGIC set.seed(412)
# MAGIC assignment <- sample(1:3, size = n, prob = c(0.7,0.15,0.15), replace = TRUE)
# MAGIC 
# MAGIC # Create a train, validation and tests from the original data frame 
# MAGIC neg_train <- dt_all[assignment == 1, ]    # subset to training indices only
# MAGIC neg_test <- dt_all[assignment == 2, ]  # subset to test indices only
# MAGIC neg_valid <- dt_all[assignment == 3, ]   # subset to validation indices only
# MAGIC 
# MAGIC 
# MAGIC ## 2) DEDUPLICATE neglabs and WDRS data sets/create potential pairs
# MAGIC 
# MAGIC pairs_1 <- compare.dedup(dt_neg
# MAGIC                        , exclude = c('row_id')
# MAGIC                        , blockfld = c('dob')
# MAGIC                        , strcmp = c('subname')
# MAGIC                        , strcmpfun = contains_name
# MAGIC                        )
# MAGIC 
# MAGIC pairs_2 <- compare.dedup(dt_wdrs
# MAGIC                        , exclude = c('row_id')
# MAGIC                        , blockfld = c('dob')
# MAGIC                        , strcmp = c('subname')
# MAGIC                        , strcmpfun = contains_name
# MAGIC                        )
# MAGIC 
# MAGIC ## dt_all is too big, sample from both or use compare.linkage
# MAGIC 
# MAGIC pairs_all <- compare.linkage(dt_neg, dt_wdrs
# MAGIC                              , exclude = c('row_id')
# MAGIC                              , blockfld = c('dob')
# MAGIC                              , strcmp = c('subname')
# MAGIC                              , strcmpfun = contains_name
# MAGIC                              )
# MAGIC 
# MAGIC 
# MAGIC dt_all_s <- sample_n(dt_all, 500000, replace = FALSE) ## ----> create sample of full dataset WITHOUT replacement
# MAGIC 
# MAGIC pairs_all <- compare.dedup(dt_all_s
# MAGIC                           , exclude = c('row_id')
# MAGIC                           , blockfld = c('dob')
# MAGIC                           , strcmp = c('subname')
# MAGIC                           , strcmpfun = contains_name
# MAGIC                           )
# MAGIC 
# MAGIC ## use old dataset for comparison
# MAGIC 
# MAGIC pairs_old <- compare.dedup(dt_train_old_1
# MAGIC                            , exclude = c('row_id')
# MAGIC                            , blockfld = c('dob')
# MAGIC                            , strcmp = c('subname')
# MAGIC                            , strcmpfun = contains_name
# MAGIC                           )
# MAGIC 
# MAGIC 
# MAGIC ## compare outputs
# MAGIC 
# MAGIC pairs_1$frequencies
# MAGIC pairs_2$frequencies
# MAGIC pairs_all$frequencies
# MAGIC 
# MAGIC pairs_1_a <- pairs_1$pairs
# MAGIC pairs_2_a <- pairs_2$pairs
# MAGIC pairs_3_a <- pairs_all$pairs
# MAGIC pairs_old_a <- pairs_old$pairs
# MAGIC 
# MAGIC ## 3) GENERATE PAIRS for updated model - old method uses "fsWeights" function; also testing the "emWeights" and "epiWeights" functions
# MAGIC 
# MAGIC pairs_w_fs <- fsWeights(pairs_all)
# MAGIC pairs_w_em <- emWeights(pairs_all)
# MAGIC pairs_w_epi <- epiWeights(pairs_all)
# MAGIC 
# MAGIC summary(pairs_w_fs)
# MAGIC summary(pairs_w_em)
# MAGIC summary(pairs_w_epi)
# MAGIC 
# MAGIC 
# MAGIC ## Extract a training set (approach #3) - uses the RecordLinkage package "getMinimalTrain" function; produces a smaller training set, but this 
# MAGIC ## is the process that was used in the original model; must be run AFTER pairs are weighted;
# MAGIC ## extracts a subset of the comparison patterns so that every pattern is represented at least once
# MAGIC 
# MAGIC trainset <- getMinimalTrain(pairs_all, nEx = 4)
# MAGIC 
# MAGIC ## copy, save, and load the training set as needed
# MAGIC ##
# MAGIC ## trainedset <- trainset
# MAGIC trainedset <- editMatch(trainedset)
# MAGIC save(trainedset, file = "c:/data/trainedset.RData")
# MAGIC ##
# MAGIC ## train a model, and save it:
# MAGIC neglabs_model <- trainSupv(trainedset, method = "bagging")
# MAGIC save(neglabs_model, file = "c:/data/neglabs_model.RData")
# MAGIC 
# MAGIC 
# MAGIC neglabs_tolink <- copy(dt_neg)
# MAGIC neglabs_tolink$subname[is.na(neglabs_tolink$subname)] <- "null"
# MAGIC table(is.na(neglabs_tolink$subname))
# MAGIC 
# MAGIC 
# MAGIC ## Compute probabilistic weights, using the Fellegi-Sunter algorithm; fsWeights calculates matching weights on an object based on the specified m- and u- probabilities;
# MAGIC 
# MAGIC pairs_1 <- compare.dedup(neglabs_tolink
# MAGIC                        , exclude = c('welrs_obx_id')
# MAGIC                        , blockfld = c('dob')
# MAGIC                        , strcmp = c('subname')
# MAGIC                        , strcmpfun = contains_name
# MAGIC                       )
# MAGIC 
# MAGIC 
# MAGIC pairs_2 <- compare.dedup(neglabs_tolink
# MAGIC                        , exclude = c('welrs_obx_id')
# MAGIC                        , blockfld = c('dob')
# MAGIC                        , strcmp = c('subname')
# MAGIC                        , strcmpfun = levenshteinSim
# MAGIC                       )
# MAGIC 
# MAGIC ## Use LevneshteinSim version
# MAGIC 
# MAGIC pairs_t <- fsWeights(pairs_2)
# MAGIC 
# MAGIC 
# MAGIC ## Create a training set
# MAGIC 
# MAGIC trainset <- getMinimalTrain(pairs_t, nEx = 4)
# MAGIC 
# MAGIC ## copy, save, and load the training set as needed
# MAGIC 
# MAGIC ## trainedset <- trainset
# MAGIC 
# MAGIC trainedset <- editMatch(trainedset)
# MAGIC save(trainedset, file = "c:/data/trainedset.RData")
# MAGIC 
# MAGIC ## Train the model
# MAGIC 
# MAGIC neglabs_model <- trainSupv(trainedset, method = "bagging")
# MAGIC save(neglabs_model, file = "c:/data/neglabs_model.RData")
