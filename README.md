# CALF v1.18.2 README


# Setup and Usage
You can install CALF by running `pip install calfpy`

You can import all CALF functions by using the folliwng import statement  
`import calfpy.methods`

Calling a function (example)
`calfpy.methods.calf(data, 3, "binary", optimize = 'pval', verbose = False)`  

# Library Documentation

## *calf(data, nMarkers, targetVector, optimize = 'pval', verbose = False)*  

Coarse Approximation Linear function.  The main function used to invoke the CALF algorithm on a dataset.  

parameter | description | type
--------- | ------------| --------
data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary | *pandas DataFrame*
nMarkers | Maximum number of markers to include in creation of sum. | *int*
targetVector | Set to "binary" to indicate data with a target vector (first column) having case/control characteristic.  Set to "nonbinary" for target vector (first column) with real numbers. | *string*
optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
verbose| True to print activity at each iteration to console. Defaults to False. | *bool*

#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
selection | The markers selected with their assigned weight (-1 or 1).
auc | The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC.
randomize | False
proportion | Undefined
targetVec | Target vector argument given in the function call, 'rocPlot': Receiver operating curve plot, if applicable for dataset type and optimizer supplied. 
finalBest | The optimal value for the provided optimization type, e.g. if optimize='pval" this will have the calculated p-value for the run.
optimize | The optimizer argument given in the function call.  



## *calf_fractional(data, nMarkers, controlProportion = .8, caseProportion = .8, optimize = "pval", verbose = False)* 

Randomly selects from binary input provided to data parameter while ensuring the requested proportions of case and control variables are used and runs Coarse Approximation Linear Function  

parameter | description | type
--------- | ------------| --------
data | Data frame where first column contains case/control coded variables (0/1) (binary data). | *pandas DataFrame*
nMarkers | Maximum number of markers to include in creation of sum. | *int*
controlProportion | Proportion of control samples to use, default is .8. | *float*
caseProportion | Proportion of case samples to use, default is .8. | *float*
optimize | Criteria to optimize.  Allowed values are "pval" or "auc". | *string*
verbose | True to print activity at each iteration to console. Defaults to False.| *bool*
#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
selection | The markers selected each with their assigned weight (-1 or 1).
auc |  The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC.
randomize | False
proportion | The proportions of case an control applied druing the function run.
targetVec | "binary"
rocPlot | Receiver operating curve plot, if applicable for dataset type and optimizer supplied.
finalBest | The optimal value for the provided optimization type, e.g. if optimize='pval" this will have the calculated p-value for the run.
optimize | The optimizer argument given in the function call.



## *calf_randomize(data, nMarkers, targetVector, times = 1, optimize = "pval", verbose = False)*

Randomly selects from input provided to data parameter and runs Coarse Approximation Linear Function  

parameter | description | type
--------- | ------------| --------
 data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary | *pandas DataFrame*
nMarkers | Maximum number of markers to include in creation of sum. | *int*
times | Indicates the number of replications to run with randomization. | *int*
optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
verbose | True to print activity at each iteration to console. Defaults to False.| *bool*
#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
multiple | The markers chosen and the number of times they were selected per iteration,
auc | The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC,
randomize | True
targetVec | "binary"
aucHist | A historgram of the AUC values calcuted for all the iterations,
times | The value provided to the times parameter when the function was called,
rocPlot | Receiver operating curve plot, if applicable for dataset type and optimizer supplied, 
finalBest | The optimal value for the provided optimization type, e.g. if optimize='pval" this will have the calculated p-value for the run,
optimize | The optimizer argument given in the function call,
verbose | The value supplied to the verbose parameter when the function was called



## *calf_subset (data, nMarkers, targetVector, proportion = .8, times = 1, optimize = "pval", verbose = False)*
Randomly selects a subset of the data on which to run Coarse Approximation Linear Function  

parameter | description | type
--------- | ------------| --------
param data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary | *pandas DataFrame*
param nMarkers | Maximum number of markers to include in creation of sum. | *int*
param targetVector | Set to "binary" to indicate data with a target vector (first column) having case/control characteristic.  Set to "nonbinary" for target vector (first column) with real numbers. | *string*
param proportion | A value between 0 and 1, the percentage of data, randomly chosen, to use in the calculation.  Default is .8. | *float*
param times | Indicates the number of replications to run with randomization. | *int*
param optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
param verbose | True to print activity at each iteration to console. Defaults to False. | *bool*
#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
multiple | The markers chosen and the number of times they were selected per iteration.
auc | The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC.
proportion | The value supplied to the proportion paremeter when calling the function.
targetVec | "binary"
aucHist | A historgram of the AUC values calcuted for all the iterations.
times | The value provided to the times parameter when the function was called.
rocPlot | Receiver operating curve plot, if applicable for dataset type and optimizer supplied.
finalBest | The optimal value for the provided optimization type, e.g. if optimize=pval" this will have the calculated p-value for the run.
optimize | The optimizer argument given in the function call.



## *calf_exact_binary_subset(data, nMarkers, nCase, nControl, times = 1, optimize = "pval", verbose = False)*
Randomly selects subsets of data, case and control, from a binary data set, while precisely ensuring the size of the sets on which to run Coarse Approximation Linear Function  

parameter | description | type
--------- | ------------| --------
data | Data frame where first column contains case/control coded variable (0/1). | *pandas DataFrame*
nMarkers | Maximum number of markers to include in creation of sum. | *int*
nCase | The number of data points to use for the set of case samples. | *int*
nControl | The number of data points to use for the set of control samples. | *int*
times | Indicates the number of replications to run with randomization | *int*
optimize | Criteria to optimize.  Allowed values are "pval" or "auc" | *string*
verbose | True to print activity at each iteration to console. Defaults to False. | *bool*
#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
multiple | The markers chosen and the number of times they were selected per iteration.
auc | The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC.
proportion | The value supplied to the proportion paremeter when calling the function.
targetVec | "binary"
aucHist | A historgram of the AUC values calcuted for all the iterations.
times | The value provided to the times parameter when the function was called.
rocPlot | Receiver operating curve plot, if applicable for dataset type and optimizer supplied.
finalBest | The optimal value for the provided optimization type, e.g. if optimize='pval" this will have the calculated p-value for the run.
optimize | The optimizer argument given in the function call.



## *calf_cv(data = CaseControl, limit = 5, times = 100, targetVector = 'binary', optimize = 'pval')*
Performs repeated random subsampling cross validation on data for Coarse Approximation Linear Function  

parameter | description | type
--------- | ------------| --------
 data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary | *pandas DataFrame*
 limit | Maximum number of markers to attempt to determine per iteration. | *int*
 times | Indicates the number of replications to run with randomization. | *int*
 proportion | A value between 0 and 1, the percentage of data, randomly chosen, to use in each iteration of CALF.  Default is .8, | *float*
 optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
 outputPath | The path where files are to be written as output, default is None meaning no files will be written.  When targetVector is "binary" file binary.csv will be output in the provided path, showing the reults.  When targetVector is "nonbinary" file nonbinary.csv will be output in the provided path, showing the results.  In the same path, the kept and excluded variables from the LAST iteration, will be output, prefixed with the targetVector type "binary" or "nonbinary" followed by Kept and Excluded and suffixed with .csv.  Two files containing the results from each run have List in the filenames and suffixed with .txt. | *string*
#### **returns**

A data frame of the results from the cross validation.  Columns of all markers from data and rows representing each iteration of a CALF run.  Cells will contain the result from CALF for a given CALF run and the markers that were chose for that run.


## *perm_target_cv(data, targetVector, limit, times, proportion = .8, optimize = 'pval', outputPath=None)*
Performs repeated random subsampling cross validation on data but randomly permutes the target column (first column) with each iteration, for Coarse Approximation Linear Function   

parameter | description | type
--------- | ------------| --------
data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary values if targetVector is nonbinary | *pandas DataFrame*
limit | Maximum number of markers to attempt to determine per iteration. | *int*
times | Indicates the number of replications to run with randomization. | *int*
proportion | A value between 0 and 1, the percentage of data, randomly chosen, to use in each iteration of CALF.  Default is .8, | *float*
optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
outputPath | The path where files are to be written as output, default is None meaning no files will be written.  When targetVector is "binary" file binary.csv will be output in the provided path, showing the reults.  When targetVector is "nonbinary" file nonbinary.csv will be output in the provided path, showing the results.  In the same path, the kept and excluded variables from the LAST iteration, will be output, prefixed with the targetVector type "binary" or "nonbinary" followed by Kept and Excluded and suffixed with .csv.  Two files containing the results from each run have List in the filenames and suffixed with .txt. | *string*
#### **returns**

A data frame of the results from the cross validation.  Columns of all markers from data and rows representing each iteration of a CALF run.  Cells will contain the result from CALF for a given CALF run and the markers that were chose for that run.


## *write_calf(x, filename)*
Writes the results from a call to calf() to a file  

parameter | description | type
--------- | ------------| --------
x | The dictionary object returned from calling calf(). | *dict*
filename | The name of the file in which to write the results from calf(). | *string*


## *write_calf_randomize(x, filename)*
Writes the results from a call to calf_randomize() to a file  

parameter | description | type
--------- | ------------| --------
x | The dictionary object returned from calling calf_randomize(). |  *dict*
filename | The name of the file in which to write the results from calf_randomize(). | *string*

	

## *write_calf_subset(x, filename)*
Writes the results from a call to calf_subset() to a file  

parameter | description | type
--------- | ------------| --------
x | The dictionary object returned from calling calf_subset(). | *dict*
filename | The name of the file in which to write the results from calf_subset(). | *string*





## *calf_internal(data, nMarkers, randomize = False, proportion = None, times = 1, targetVector = 'binary', optimize = 'pval', verbose = False)*
The basic CALF algorithm  

parameter | description | type
--------- | ------------| --------
data | Data frame where first column contains case/control coded variable (0/1) if targetVector is binary or real values if targetVector is nonbinary | *pandas DataFrame*
nMarkers | Maximum number of markers to include in creation of sum. | *int*
randomize | Set to True to randomize the data for each CALF run. | *bool*
proportion | A value between 0 and 1, the percentage of data, randomly chosen, to use in the calculation. | *float*
times | The number of times to run CALF on data. | *int*
targetVector | Set to "binary" to indicate data with a target vector (first column) having case/control characteristic.  Set to "nonbinary" for target vector (first column) with real numbers. | *string*
optimize | Criteria to optimize.  Allowed values are "pval", "auc", for a binary target vector, or "corr" for a nonbinary target vector. | *string*
verbose | True to print activity at each iteration to console. Defaults to False. | * bool*
#### **returns**

A dictionary composed of the following results from CALF:

key | description/value
--- | -----------
selection | The markers selected each with their assigned weight (-1 or 1).
auc | The AUC determined during running CALF.  AUC can be provided for given markers AUC represented for selected markers will only be optimal if set to optimzie for AUC.
randomize | False
proportion | Undefined
targetVec | Target vector argument given in the function call.
rocPlot | Receiver operating curve plot, if applicable for dataset type and optimizer supplied.
finalBest | The optimal value for the provided optimization type, e.g. if optimize='pval" this will have the calculated p-value for the run.
optimize | The optimizer argument given in the function call.