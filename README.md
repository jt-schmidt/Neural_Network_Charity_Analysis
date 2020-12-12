# Neural_Network_Charity_Analysis
Module 19 UT Data Visualization Bootcamp -- Neural Network Exploration using Python &amp; Jupyter Notebook

<!-- The purpose of this analysis is well defined (4 pt) -->
## Overview

Module 19 challenge assignment is broken in to 3 specific deliverables:
1.  Pre-process sample data set for use within a neural network model
    * Includes column value grouping based on uniqueness
2.  Compile, Train, and Evaluate Model using [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
    * Includes initial determination of neuron count within hidden node layers
3.  Optimization of the Model with goal to achieve target predictive accuracy higher than 75%
    * This was exercise on exploring effects on model accuracy

Sample Data
* Over 34,000 organizations
* Variety of metadata for each organization
  * EIN and NAME—Identification columns
  * APPLICATION_TYPE—Alphabet Soup application type
  * AFFILIATION—Affiliated sector of industry
  * CLASSIFICATION—Government organization classification
  * USE_CASE—Use case for funding
  * ORGANIZATION—Organization type
  * STATUS—Active status
  * INCOME_AMT—Income classification
  * SPECIAL_CONSIDERATIONS—Special consideration for application
  * ASK_AMT—Funding amount requested
  * IS_SUCCESSFUL—Was the money used effectively
  
Resources used include:
  * Python in Jupyter Notebook
  * Tensorflow
  * Pandas
  * Sklearn

<!-- There is a bulleted list that answers all six questions (15 pt) -->
## Results

* Data Preprocessing
  
  1. What variable(s) are considered the target(s) for your model?
      
      IS_SUCCESSFUL column is the target or output of the model.

  2. What variable(s) are considered to be the features for your model?
      
      Generally, all other columns in dataframe besides the "target" would be considered "features".  Features are those variables which act as an input to affect output.  
      
      ``` Python
         # Split our preprocessed data into our features and target arrays
         x = app_merge_df.drop(columns="IS_SUCCESSFUL").values
         y = app_merge_df.IS_SUCCESSFUL.values
      ```
      
      For this exercise, the features are:
      * STATUS
      * ASK_AMT
      * IS_SUCCESSFUL
      * APPLICATION_TYPE_Other
      * APPLICATION_TYPE_T10
      * APPLICATION_TYPE_T19
      * APPLICATION_TYPE_T3
      * APPLICATION_TYPE_T4
      * APPLICATION_TYPE_T5
      * APPLICATION_TYPE_T6
      * APPLICATION_TYPE_T7
      * APPLICATION_TYPE_T8
      * AFFILIATION_CompanySponsored
      * AFFILIATION_Family/Parent
      * AFFILIATION_Independent
      * AFFILIATION_National
      * AFFILIATION_Other
      * AFFILIATION_Regional
      * CLASSIFICATION_C1000
      * CLASSIFICATION_C1200
      * CLASSIFICATION_C2000
      * CLASSIFICATION_C2100
      * CLASSIFICATION_C3000
      * CLASSIFICATION_Other
      * USE_CASE_CommunityServ
      * USE_CASE_Heathcare
      * USE_CASE_Other
      * USE_CASE_Preservation
      * USE_CASE_ProductDev
      * ORGANIZATION_Association
      * ORGANIZATION_Co-operative
      * ORGANIZATION_Corporation
      * ORGANIZATION_Trust
      * INCOME_AMT_0
      * INCOME_AMT_1-9999
      * INCOME_AMT_10000-24999
      * INCOME_AMT_100000-499999
      * INCOME_AMT_10M-50M
      * INCOME_AMT_1M-5M
      * INCOME_AMT_25000-99999
      * INCOME_AMT_50M+
      * INCOME_AMT_5M-10M
      * SPECIAL_CONSIDERATIONS_N
      * SPECIAL_CONSIDERATIONS_Y
      

  3. What variable(s) are neither targets nor features, and should be removed from the input data?
  
  EIN & NAME columns are not targets or featurs and were removed from the model.
  
  ``` Python
   # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
   app_modify_df = application_df.drop(['EIN','NAME'], axis=1)
   app_modify_df.head()
  ```

* Compiling, Training, and Evaluating the Model
  
  4. How many neurons, layers, and activation functions did you select for your neural network model, and why?
  
      For Neurons, I worked from 2x the number of features being used & also doubling for each layer which existed.
      * Features = 43
      * Layer 1 = 172 = 43 * 2 * 2
      * Layer 2 = 86 = 43 * 2
      * Layer 3 = 43 
      These values were arrived at by estimation and review of articles such as shown below.
  
      For reference:
      * https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
      * https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw#:~:text=The%20number%20of%20hidden%20neurons,size%20of%20the%20input%20layer.

      For activation functions, "relu" and "swish" were used.
         * "relu" was primarily used by default since it had been used throughout most of the module examples
         * "swish" was alternatively chosen as something new to apply based on options found in [Tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/activations).

  5. Were you able to achieve the target model performance?
      
      Unfortunately, no.  
      Highest achieved accuracy was during Attempt 2:  0.7259474992752075
      This was done by doubling epoch count from 50 to 100 and doubling neurons in the hidden layers.
  
  6. What steps did you take to try and increase model performance?
      
      * Increase Epoch count from 50 to 100
      * Doubling neuron count within hidden layers
      * Adding a 3rd hidden layer
      * Changing activation from "relu" to "swish"

<!-- Summarize the overall results of the deep learning model. 
Include a recommendation for how a different model could solve this classification problem, and explain your recommendation. 
There is a summary of the results (2 pt)
There is a recommendation on using a different model to solve the classification problem, and justification (3 pt)-->
## Summary

Overall, this was a good exercise in exploration of neural network machine learning using Python & Tensorflow.  It took through standard steps of data review, clean-up or pre-processing, variable selection for input/output, model creation, and evaluation.

Then it allowed more freeform exploration of how machine learning model could be optimized to achieve higher overall accuracy.

As a recommendation to further improve accuracy, I would consider an application which can put X & Y variables through a variety of standard models with base / default parameter selection.  Resultant output would compare accuracy for each model.  Based on model selected, then further refinement could be explored by modifying default parameters and / or modifying column dataset.
