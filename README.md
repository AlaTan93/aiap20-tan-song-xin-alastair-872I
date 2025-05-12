# aiap20-tan-song-xin-alastair-872I

## Execution
Python version used is 3.13.3. Install all requisites from requirements.txt

To execute, execute run.sh in a Linux environment with the argument "train" or the argument "eval". Examples are:

sh run train  
OR  
sh run eval  

## Data Preprocessing:

* Client ID: dropped as the field will only cause overfitting.
* Age: Inputted into the database as a string styled like '15 years'. In addition, a significant portion of the users have their age set as '150 years', which is a technical impossibility. The data is converted to its integer equivalents, and the records with '150 years' are imputed to become other values. Other than that, the data follows a Gaussian distribution.
* Occupation: categorical. Some categories are condensed:
    * blue-collar: consisting of those categorised as 'technician', 'housemaid', 'blue-collar'
    * white-collar: consisting of those categorised as 'admin.', 'management', 'services'
    * self-employed: consisting of those categorised as 'entrepreneur' and 'self-employed'
    * Students, the retired, unemployed, and those marked as unknown remain the same
