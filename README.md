# aiap20-tan-song-xin-alastair-872I

## Execution
Python version used is 3.13.3. Install all requisites from requirements.txt

To execute, execute run.sh in a Linux environment with the argument "train" or the argument "eval", or without an argument to run in training mode. **eval** only works if there are saved pipelines from previous runs of **train**, and only with a test_data.csv file in the same folder directory as run.sh, with the same structure as the **bmarket.db** file.

Examples are:

sh run train  
OR  
sh run eval  

## Data Preprocessing:

* Client ID: dropped as the field will only cause overfitting.
* Age: Inputted into the database as a string styled like '15 years'. In addition, a significant portion of the users have their age set as '150 years', which is a technical impossibility. The data is converted to its integer equivalents, and the records with '150 years' are imputed to become other values. Other than that, the data follows a Gaussian distribution.
* Occupation: categorical. Encoded as one-hot. Some categories are condensed:
    * blue-collar: consisting of those categorised as 'technician', 'housemaid', 'blue-collar'
    * white-collar: consisting of those categorised as 'admin.', 'management', 'services'
    * self-employed: consisting of those categorised as 'entrepreneur' and 'self-employed'
    * Students, the retired, unemployed, and those marked as unknown remain the same
* Education Level: categorical. Encoded as one-hot. One category is condensed out ouf all of them:
    * high.school.ab: "High school and below". Consists of everyone with high school education, basic education, or illiterate
    * The rest of the categories are kept as is.
* Marital Status: categorical. Encoded as one-hot.
* Credit Default: categorical. Few 'yes', many 'no's, some unknowns. Dropped as hypothesised as the data may cause records to overfit.
* Housing Loan: categorical. Around 50% of records are marked as 'None'. Will reclassify them as 'unknown', a record type that already exists. One-hot encoding is done.
* Personal Loan: categorical. Around 10% of records are marked as 'None'. Will reclassify them as 'unknown', a record type that already exists. One-hot encoding is done.
* Contact Method: categorical. Data is dirty, ultimately collated to two categories 'celullar' and 'telephone. One-hot encoding is done.
* Campaign Calls: numerical. Some negative values. All values converted to their absolute value. Distribution follows roughly the 'right half' of a normal distribution. A StandardScaler is used afterwards as well.
* Previous Contact Days: numerical. Some values set as 999. Having the values set as 999 will skew results when scaling. Values set as 999 are changed to 0, and an additional column is engineered called "Contact Days Mask"
* Contact Days Mask: boolean, feature engineered. Set as 1 if the Previous Contact Days value was previously 999, set as 0 otherwise. Serves as a 'data mask' that is hopefully used by any trained models.
* Subscription Status: boolean. Label.
    * For the DecisionTreeRegressor, and the Torch Deep Model, this is changed to a float.
        * The DecisionTreeRegressor uses 0.05 to represent 0, and 1.0 to represent 1
        * The Deep Model uses 0.05 to represent 0, and 0.9 to represent 1
        * Soft labels are expected to help reduce overfitting, at the cost of additional pipeline complexity.
    * The XGBoost Classifier uses 0 and 1 as per normal

## Algorithms / Models

A 20% validation split is created, then the remaining 80% is used in 5-Fold Cross-Validation to compare all the models. As the data is skewed having more '0' than '1' labels, SMOTE is used to oversample the datasets for all pipelines. The three models used are:

* Decision Tree Regressor. A Decision Tree Regressor was chosen over a Decision Tree Classifier as the first model as a Classifier cannot be used on a soft labelling method. Plus, a decision tree model is easy to explain, even if prone to overfitting.
    * Uses soft labels of 0.05 and 1.0
    * The parameters searched using GridSearchCV are:
        * "max_depth": [3, 5, 6, 7, 8, 9, 10, None]
        * "model__min_samples_split": [2, 5, 10, 20, 50, 100]
    * A custom Binary Cross Entropy Loss was created as the scorer, treating values <0.5 as class 0, and values >= 0.5 as class 1.
* XGBoost Classifier. No soft labels are used. This model was chosen, expecting as a middle ground between deep models, and decision tree models.
    * The parameters searched using GridSearchCV are:
        * n_estimators": [300, 600, 900]
        * max_depth":    [3, 4, 6, None]
        * learning_rate":[0.03, 0.05, 0.1]
        * subsample":    [0.6, 0.8, 1.0]
        * colsample_bytree":[0.6, 0.8, 1.0]
    * F1 was used as the scorer.
* A Deep Model using PyTorch. It was believed that a deep model can capture the non-linear relationships better than the other models.
    * Uses soft labels of 0.05 and 0.9
    * The network is as structured:
        * Linear(27, 64) + BatchNorm1d + GELU
        * Linear(64, 100) + BatchNorm1d + GELU
        * Linear(100, 128) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(128, 148) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(148, 190) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(190, 270) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(270, 250) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(250, 180) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(180, 120) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(120, 64) + BatchNorm1d + GELU + Dropout(0.1)
        * Linear(64, 1)
    * The parameters are:
        * Learning_rate = 1e-6
        * Max Epochs = 100
        * Batch Size = 900
        * Optimizer: Adam
        * Loss: BCE With Logits Loss
        * Early-Stopping: Yes
        * Patience: 5
        * Minimum Validation Loss Difference for Patience: 1e-4
    
## Evaluation Metrics:

* F1 Score is the best option for an evaluation metric, as the bank likely values positives, and the reduction of false positives is most intended.
* Binary Cross-Entropy Loss is an acceptable metric for a training loss.
* Balanced Accuracy penalises being more accurate for 1 label than another, but more tolerant of false positives than F1.

## Components in Pipeline:
* The pipeline consists of:
    * A StandardScaler, scaling only numerical columns.
    * (For the Deep Model) A layer to convert all input columns not integers into float32s, for better training in torch
    * SMOTE, meant to perform synthetic oversampling.
    * (For the Deep Model and the Decision Tree Regressor) A layer to convert hard labels (0, 1) to soft labels of different values for each training model
    * The model.

## Results:
    * DecisionTreeRegressor:
        * Best parameters: max_depth: None, min_samples_split: 100
        * Testing using the fold model with the best results:
            * Validation Hold‑out BCE: 0.4761
            * Validation Hold‑out F1: 0.3044
            * Validation Hold‑out Balanced Accuracy: 0.6200
        * Complete Dataset Validation:
            * BCE: 0.4057
            * F1: 0.3790
            * Balanced Accuracy: 0.6709
    * XGBoost Classifier:
        * Best parameters: colsample_bytree: 0.8, learning_rate: 0.03, max_depth: 4, n_estimators: 600, subsample: 0.8
        * Testing using the fold model with the best results:
            * Validation Hold‑out F1: 0.3514
            * Validation Hold‑out Balanced Accuracy: 0.6321
        * Complete Dataset Validation:
            * F1: 0.3652
            * Balanced Accuracy: 0.6382
    * Deep Model:
        * Only 5 Fold-Validation was done.
            * Validation Hold-out Hard BCE: 0.5728
            * Validation Hold-out F1: 0.3252
            * Validation Hold-out Balanced Accuracy: 0.6487
        * Complete Dataset Validation:
            * BCE: 1.0325
            * F1: 0.3335
            * Balanced Accuracy: 0.6067

## Reflection
The performance of the deep model is disappointing, despite the depth of the model created. It is possible, even likely, that a stringent and heavy search of all possible permutations and parameters will discover a better model than the XGBoost Classifier, which has the highest validation hold-out F1 Score.

The DecisionTreeRegressor has the best performance on the whole dataset, but weaker than XGBoostClassifier on the holdout. The Deep model performed the worst out of all 3 in terms of F1, but had better balanced accuracy than the XGBoostClassifier. A custom loss option to prefer true positives and discourage false positives for the Deep Model should be considered in the future.

Using the best model structure and performing a full retrain was considered, but ultimately not done due to a lack of time.

This project came at an inconvenient time, as two final exams for two Masters-level courses were scheduled on the same week. Ultimately, only 1.5 days were available for this project.
