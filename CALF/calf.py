from plotnine import *
from scipy.stats import ttest_ind
import logging
import math
import numpy
import os
import pandas
import warnings


#Basic Configuration for warnings from plotnine and logging.
warnings.filterwarnings('ignore', module='plotnine')


log = logging.getLogger(__name__)


def computeAuc(caseVar, ctrlVar):
	funcValue = caseVar.tolist() + ctrlVar.tolist()
	rounded = [round(element, 8) for element in funcValue]
	ranks = pandas.Series.rank(pandas.Series(rounded)).tolist()
	caseFunc = sum(ranks[0:len(caseVar)]) - len(caseVar)*(len(caseVar)+1)/2
	ctrlFunc = sum(ranks[(len(caseVar)):len(ranks)]) - len(ctrlVar)*(len(ctrlVar)+1)/2
	return round(max(ctrlFunc, caseFunc)/(caseFunc + ctrlFunc),4)



def calf(data, nMarkers, targetVector, optimize = 'pval', verbose = False):
#'@title calf
#'@description Coarse Approximation Linear Function
#'@param data Matrix or data frame. First column must contain case/control dummy coded variable (if targetVector = "binary"). Otherwise, first column must contain real number vector corresponding to selection variable (if targetVector = "nonbinary"). All other columns contain relevant markers.
#'@param nMarkers Maximum number of markers to include in creation of sum.
#'@param targetVector Indicate "binary" for target vector with two options (e.g., case/control). Indicate "nonbinary" for target vector with real numbers.
#'@param optimize Criteria to optimize, "pval" or "auc", (if targetVector = "binary") or "corr" (if targetVector = "nonbinary").  Defaults to "pval".
#'@param verbose Logical. Indicate True to print activity at each iteration to console. Defaults to False.
#'@return A data frame containing the chosen markers and their assigned weight (-1 or 1)
#'@return The optimal AUC, pval, or correlation for the classification.
#'@return If targetVector is binary, rocPlot. A plot object from ggplot2 for the receiver operating curve.
#'@examples
#'calf(data = CaseControl, nMarkers = 6, targetVector = "binary", optimize = "pval")
#'@export

	if targetVector != 'binary' and targetVector != 'nonbinary':
		raise Exception('CALF ERROR: Invalid targetVector argument.  Only "binary" or "nonbinary" is allowed.')
	elif targetVector == 'binary' and optimize=='corr':
		raise Exception('CALF ERROR: Optimizing by "corr" is only applicable to nonbinary data.')
	elif targetVector == 'nonbinary' and optimize=='pval':
		raise Exception('CALF ERROR: Optimizing by "pval" is only applicable to binary data.')
	elif targetVector == 'nonbinary' and optimize=='auc':
		raise Exception('CALF ERROR: Optimizing by "auc" is only applicable to binary data.')

	result = calf_internal(data,
				  nMarkers,
				  proportion = None,
				  randomize  = False,
				  targetVector = targetVector,
				  times      = 1,
				  optimize = optimize,
				  verbose = verbose)

	if result['randomize'] == True:
		print('Randomized Output:\n\n')

	if result['proportion'] is not None:
		print('Proportion of Data:' + str(result.proportion) + '\n\n')

	print(result['selection'].to_string(index=False))
	print()
	print_optimizer(result)
	
	print('\n')
	return result
	

def calf_fractional(data, nMarkers, controlProportion = .8, caseProportion = .8, optimize = "pval", verbose = False):
#'@title calf_fractional
#'@description Randomly selects from binary input provided to data parameter while ensuring the requested proportions of case and control variables are used and runs Coarse Approximation Linear Function.
#'@param data Matrix or data frame. Must be binary data such that the first column must contain case/control dummy coded variable, as function is only approprite for binary data.
#'@param nMarkers Maximum number of markers to include in creation of sum.
#'@param controlProportion Proportion of control samples to use, default is .8.
#'@param caseProportion Proportion of case samples to use, default is .8.
#'@param optimize Criteria to optimize, "pval" or "auc".  Defaults to "pval".
#'@param verbose Logical. Indicate True to print activity at each iteration to console. Defaults to False.
#'@return A data frame containing the chosen markers and their assigned weight (-1 or 1)
#'@return The optimal AUC or pval for the classification.
#'@return rocPlot. A plot object from ggplot2 for the receiver operating curve.
#'@examples
#'calf_fractional(data = CaseControl, nMarkers = 6, controlProportion = .8, caseProportion = .4)
#'@export

	if optimize != 'pval' and optimize != 'auc':
		raise Exception('CALF ERROR: calf_fractional is only applicable to binary datasets.  Options for parameter "optimize" are "pval" or "auc".')
	elif controlProportion is None and caseProportion is None:
		raise Exception('CALF ERROR: "caseProportion" and/or "controlProportion" cannot be None')
	elif controlProportion <= 0 or controlProportion >= 1:
		raise Exception('CALF ERROR: value for "controlProportion" must be between 0.0 and 1.0')
	elif caseProportion <= 0 or caseProportion >= 1:
		raise Exception('CALF ERROR: value for "caseProportion" must be between 0.0 and 1.0')
	elif caseProportion + controlProportion > 1:
		raise Exception('CALF ERROR: parameters "caseProportion" and "controlProportion" must sum to 1.0')

	result = calf_internal(data,
				  nMarkers,
				  proportion = [controlProportion,caseProportion],
				  randomize = False,
				  targetVector = 'binary',
				  times = 1,
				  optimize = optimize,
				  verbose = verbose)
				  
	if result['randomize'] == True:
		print('Randomized Output:\n\n')

	if result['proportion'] is not None:
		print('Proportion of Data: {}\n\n'.format(result['proportion']))

	print(result['selection'])
	print()
	print_optimizer(result)
	
	print('\n')


#'@title calf_randomize
#'@description Randomly selects from binary input provided to data parameter and runs Coarse Approximation Linear Function.
#'@param data Matrix or data frame. Must be binary data such that the first column must contain case/control dummy coded variable, as function is only approprite for binary data.
#'@param nMarkers Maximum number of markers to include in creation of sum.
#'@param targetVector Indicate "binary" for target vector with two options (e.g., case/control). Indicate "nonbinary" for target vector with real numbers.
#'@param times Numeric. Indicates the number of replications to run with randomization.
#'@param optimize Criteria to optimize if targetVector = "binary." Indicate "pval" to optimize the p-value corresponding to the t-test distinguishing case and control. Indicate "auc" to optimize the AUC.
#'@param verbose Logical. Indicate True to print activity at each iteration to console. Defaults to False.
#'@return A data frame containing the chosen markers and their assigned weight (-1 or 1)
#'@return The optimal AUC, pval, or correlation for the classification.
#'@return aucHist A histogram of the AUCs across replications, if applicable.
#'@examples
#'calf_randomize(data = CaseControl, nMarkers = 6, targetVector = "binary", times = 5)
#'@export
def calf_randomize(data,
	nMarkers,
	targetVector,
	times = 1,
	optimize = "pval",
	verbose = False):
	
	
	if targetVector != 'binary' and targetVector != 'nonbinary':
		raise Exception('CALF ERROR: Invalid targetVector argument.  Only "binary" or "nonbinary" is allowed.')
	elif targetVector == 'binary' and optimize=='corr':
		raise Exception('CALF ERROR: Optimizing by "corr" is only applicable to nonbinary data.')
	elif targetVector == 'nonbinary' and optimize=='pval':
		raise Exception('CALF ERROR: Optimizing by "pval" is only applicable to binary data.')
	elif targetVector == 'nonbinary' and optimize=='auc':
		raise Exception('CALF ERROR: Optimizing by "auc" is only applicable to binary data.')

	auc = None
	aucOut = None
	finalBestOut = None
	finalBest  = list()
	data_dict = dict()
	rocPlot = None
	aucHist = None
	
	randomize = True

	if times == 1:
	
		result = calf_internal( \
			data=data, \
			nMarkers = nMarkers, \
			proportion = None, \
			randomize = randomize, \
			targetVector = targetVector, \
			times = times, \
			optimize = optimize, \
			verbose = verbose
		)
		
		print('Randomized Output Across {} Replication:'.format(times))
		print(result['selection'].to_string(index=False))
		
		rocPlot = result['rocPlot']
		
		summaryMarkers = result['selection']
		print()
		print_optimizer(result)
		
		if targetVector == 'binary':
			if optimize == 'auc':
				aucOut = result['auc']
			else:
				finalBestOut = result['finalBest']
		else:
			finalBestOut = result['finalBest']



	else:
		count = 0
		while count < times:
			result = calf_internal( \
				data=data, \
				nMarkers = nMarkers, \
				proportion = None, \
				randomize = randomize, \
				targetVector = targetVector, \
				times = times, \
				optimize = optimize, \
				verbose = verbose
			)
			if result['auc'] is not None:
				if auc is None:
					auc = list()
				auc.append(result['auc'])

			for marker in result['selection']['Marker']:
				if marker in data_dict:
					data_dict[marker] = data_dict[marker]+1
				else:
					data_dict[marker] = 1

			finalBest.append(result['finalBest'])

			count += 1
			
		if targetVector == 'binary':
			aucPlot = pandas.DataFrame(auc)
			aucPlot.columns = ['AUC']
			aucHist = ggplot(aucPlot, aes('AUC')) + \
				geom_histogram() + \
				ylab("Count") + \
				xlab("AUC") + \
				scale_x_continuous() + \
				theme_bw()
		else:
			aucHist = None
			
		print('Randomized Output Across {} Replications: '.format(times))
		summaryMarkers = pandas.DataFrame(data_dict.items(), columns=['Markers','Frequency'])
		summaryMarkers = summaryMarkers.sort_values(by='Frequency')
		print(summaryMarkers.to_string(index=False))
		print()
		
		aucOut = None
		finalBestOut = None
		if targetVector == 'binary':
			if optimize == 'auc':
				aucOut = pandas.DataFrame(auc)
				aucOut.columns = ['AUC']
				print(aucOut.to_string(index=False))
			else:
				finalBestOut = pandas.DataFrame(finalBest)
				finalBestOut.columns = ['Final p-values']
				print(finalBestOut.to_string(index=False))
		else:
			finalBestOut = pandas.DataFrame(finalBest)
			finalBestOut.columns = ['Final Correlations']
			print(finalBestOut.to_string(index=False))
			
	if rocPlot is not None:
		print(rocPlot)
	
	print('\n')
	
	return {
		'multiple': summaryMarkers,
		'auc': aucOut,
		'randomize': randomize,
		'targetVec': targetVector,
		'aucHist': aucHist,
		'times': times,
		'finalBest': finalBestOut,
		'rocPlot': rocPlot,
		'optimize': optimize,
		'verbose': verbose
	}
	




#'@title calf_subset
#'@description Runs Coarse Approximation Linear Function on a random subset of the data provided, resulting in the same proportion applied to case and control, when applicable.
#'@param data Matrix or data frame. First column must contain case/control dummy coded variable (if targetVector = "binary"). Otherwise, first column must contain real number vector corresponding to selection variable (if targetVector = "nonbinary"). All other columns contain relevant markers.
#'@param nMarkers Maximum number of markers to include in creation of sum.
#'@param proportion Numeric. A value between 0 and 1 indicating the proportion of cases and controls to use in analysis (if targetVector = "binary"). If targetVector = "nonbinary", this is just a proportion of the full sample. Used to evaluate robustness of solution. Defaults to 0.8.
#'@param targetVector Indicate "binary" for target vector with two options (e.g., case/control). Indicate "nonbinary" for target vector with real numbers.
#'@param times Numeric. Indicates the number of replications to run with randomization.
#'@param optimize Criteria to optimize if targetVector = "binary." Indicate "pval" to optimize the p-value corresponding to the t-test distinguishing case and control. Indicate "auc" to optimize the AUC.
#'@param verbose Logical. Indicate True to print activity at each iteration to console. Defaults to False.
#'@return A data frame containing the chosen markers and their assigned weight (-1 or 1)
#'@return The optimal AUC, pval, or correlation for the classification. If multiple replications are requested, a data.frame containing all optimized values across all replications is returned.
#'@return aucHist A histogram of the AUCs across replications, if applicable.
#'@examples
#'calf_subset(data = CaseControl, nMarkers = 6, targetVector = "binary", times = 5)
#'@export

def calf_subset (data, nMarkers, targetVector, proportion = .8, times = 1, optimize = "pval", verbose = False):

	if targetVector != 'binary' and targetVector != 'nonbinary':
		raise Exception('CALF ERROR: Invalid targetVector argument.  Only "binary" or "nonbinary" is allowed.')
	elif targetVector == 'binary' and optimize=='corr':
		raise Exception('CALF ERROR: Optimizing by "corr" is only applicable to nonbinary data.')
	elif targetVector == 'nonbinary' and optimize=='pval':
		raise Exception('CALF ERROR: Optimizing by "pval" is only applicable to binary data.')
	elif targetVector == 'nonbinary' and optimize=='auc':
		raise Exception('CALF ERROR: Optimizing by "auc" is only applicable to binary data.')

	auc = None
	aucOut = None
	finalBest = list()
	finalBestOut = None
	data_dict = dict()
	rocPlot = None
	aucHist = None

	
	if times == 1:
		result = calf_internal( \
			data, \
			nMarkers, \
			proportion = proportion, \
			randomize  = False, \
			targetVector = targetVector, \
			times=times, \
			optimize = \
			optimize, \
			verbose = verbose\
		)
		
		print('Proportion = {} Output Across {} Replication:'.format(proportion, times))
		print(result['selection'].to_string(index=False))
		
		summaryMarkers = result['selection']
		print()
		print_optimizer(result)
		
		rocPlot = result['rocPlot']
	
	else:
		count = 0
		while count < times:
			result = calf_internal( \
				data, \
				nMarkers, \
				proportion = proportion, \
				randomize  = False, \
				targetVector = targetVector, \
				times=times, \
				optimize = optimize, \
				verbose = verbose \
			)
			if result['auc'] is not None:
				if auc is None:
					auc = list()
				auc.append(result['auc'])

			for marker in result['selection']['Marker']:
				if marker in data_dict:
					data_dict[marker] = data_dict[marker]+1
				else:
					data_dict[marker] = 1

			finalBest.append(result['finalBest'])

			count += 1
		
		if targetVector == 'binary':
			aucPlot = pandas.DataFrame(auc)
			aucPlot.columns = ['AUC']
			aucHist = ggplot(aucPlot, aes('AUC')) +\
			  geom_histogram() +\
			  ylab("Count") +\
			  xlab("AUC") +\
			  scale_x_continuous() +\
			  theme_bw()
		
	
		print("Proportion = " + str(proportion) + " Output Across " + str(times) + " Replications:")
		summaryMarkers = pandas.DataFrame(data_dict.items(), columns=['Markers','Frequency'])
		summaryMarkers = summaryMarkers.sort_values(by='Frequency')
		print(summaryMarkers.to_string(index=False))
		print()
		
		aucOut = None
		finalBestOut = None
		if targetVector == 'binary':
			if optimize == 'auc':
				aucOut = pandas.DataFrame(auc)
				aucOut.columns = ['AUC']
				print(aucOut.to_string(index=False))
			else:
				finalBestOut = pandas.DataFrame(finalBest)
				finalBestOut.columns = ['Final p-values']
				print(finalBestOut.to_string(index=False))
		else:
			finalBestOut = pandas.DataFrame(finalBest)
			finalBestOut.columns = ['Final Correlations']
			print(finalBestOut.to_string(index=False))

	print('\n')
	
	return {
		'multiple': summaryMarkers,
		'auc': aucOut,
		'proportion': proportion,
		'targetVec': targetVector,
		'aucHist': aucHist,
		'times': times,
		'finalBest': finalBestOut,
		'rocPlot': rocPlot,
		'optimize': optimize
	}




#'@title calf_exact_binary_subset
#'@description Runs Coarse Approximation Linear Function on a random subset of binary data provided, with the ability to precisely control the number of case and control data used.
#'@param data Matrix or data frame. First column must contain case/control dummy coded variable.
#'@param nMarkers Maximum number of markers to include in creation of sum.
#'@param nCase Numeric. A value indicating the number of case data to use.
#'@param nControl Numeric. A value indicating the number of control data to use.
#'@param times Numeric. Indicates the number of replications to run with randomization.
#'@param optimize Criteria to optimize.  Indicate "pval" to optimize the p-value corresponding to the t-test distinguishing case and control. Indicate "auc" to optimize the AUC.
#'@param verbose Logical. Indicate TRUE to print activity at each iteration to console. Defaults to FALSE.
#'@return A data frame containing the chosen markers and their assigned weight (-1 or 1)
#'@return The optimal AUC or pval for the classification. If multiple replications are requested, a data.frame containing all optimized values across all replications is returned.
#'@return aucHist A histogram of the AUCs across replications, if applicable.
#'@examples
#'calf_exact_binary_subset(data = CaseControl, nMarkers = 6, nCase = 5, nControl = 8, times = 5)
#'@export
def calf_exact_binary_subset(data, nMarkers, nCase, nControl, times = 1, optimize = "pval", verbose = False):

	auc = None
	targetVector = "binary"
	finalBest = list()
	proportion = 1
	data_dict = dict()
	rocPlot = None
	aucHist = None


	#Determine which is case and which is control
	ctrlRows = data.loc[data.iloc[:,0] == 0]
	if nControl > len(ctrlRows.index):
		raise Exception('CALF ERROR: Requested number of control rows "nControl" is larger than the number of control rows.  Please revise this value.')
	
	caseRows = data.loc[data.iloc[:,0] == 1]
	if nCase > len(caseRows.index):
		raise Exception('CALF ERROR: Requested number of case rows "nCase" is larger than the number of case rows.  Please revise this value.')
	
	if times == 1:
		
		#Resample the binary data, thus controlling the randomization here.
		ctrl = ctrlRows.sample(n=nControl)
		case = caseRows.sample(n=nCase)
		keepData  = pandas.concat([ctrl,case])

		result = calf_internal(\
			keepData,\
			nMarkers,\
			proportion = proportion,\
			randomize  = False,\
			targetVector = targetVector,\
			times = times,\
			optimize = optimize,\
			verbose = verbose\
		)
		
		
		print('Using {} out of {} case and {} out of {} control.  Output across {} replication.'.format(\
			nCase,\
			len(caseRows.index),\
			nControl,\
			len(ctrlRows.index),\
			times)\
		)
		print(result['selection'].to_string(index=False))
		print()
		print_optimizer(result)
		
		rocPlot = result['rocPlot']
		summaryMarkers = result['selection']
		

	else:
		count = 0
		while count < times:

			#Resample the binary data, thus controlling the randomization here
			#	rather than via the randomize parameter.
			ctrl = ctrlRows.sample(n=nControl)
			case = caseRows.sample(n=nCase)
			keepData  = pandas.concat([ctrl,case])

			result = calf_internal(\
				keepData,\
				nMarkers,\
				proportion = proportion,\
				randomize = False,\
				targetVector = targetVector,\
				times = times,\
				optimize = optimize,\
				verbose = verbose
			)
			
			if result['auc'] is not None:
				if auc is None:
					auc = list()
				auc.append(result['auc'])
					
					
			for marker in result['selection']['Marker']:
				if marker in data_dict:
					data_dict[marker] = data_dict[marker]+1
				else:
					data_dict[marker] = 1


			finalBest.append(result['finalBest'])
			
			count = count + 1
		
		if targetVector == 'binary':
			aucPlot = pandas.DataFrame(auc)
			aucPlot.columns = ['AUC']
			aucHist = ggplot(aucPlot, aes('AUC')) +\
			  geom_histogram() +\
			  ylab("Count") +\
			  xlab("AUC") +\
			  scale_x_continuous() +\
			  theme_bw()
		
		
		print('Using {} out of {} case and {} out of {} control.  Output across {} replications.'.format(\
			nCase,\
			len(caseRows.index),\
			nControl,\
			len(ctrlRows.index),\
			times)\
		)
		
		summaryMarkers = pandas.DataFrame(data_dict.items(), columns=['Markers','Frequency'])
		summaryMarkers = summaryMarkers.sort_values(by='Frequency')
		print(summaryMarkers.to_string(index=False))
		print()
		if targetVector == 'binary':
			if optimize == 'auc':
				aucOut = pandas.DataFrame(auc)
				aucOut.columns = ['AUC']
				print(aucOut.to_string(index=False))
			else:
				finalBestOut = pandas.DataFrame(finalBest)
				finalBestOut.columns = ['Final p-values']
				print(finalBestOut.to_string(index=False))
		else:
			finalBestOut = pandas.DataFrame(finalBest)
			finalBestOut.columns = ['Final Correlations']
			print(finalBestOut.to_string(index=False))
			
	print('\n')
	
	return {
		'multiple': summaryMarkers,
		'auc': auc,
		'proportion': proportion,
		'targetVec': targetVector,
		'aucHist': aucHist,
		'times': times,
		'finalBest': finalBest,
		'rocPlot': rocPlot,
		'optimize': optimize
	}





#'@title cv.calf
#'@description Performs cross-validation using CALF data input
#'@param data Matrix or data frame. First column must contain case/control dummy coded variable (if targetVector = "binary"). Otherwise, first column must contain real number vector corresponding to selection variable (if targetVector = "nonbinary"). All other columns contain relevant markers.
#'@param limit Maximum number of markers to include in creation of sum.
#'@param proportion Numeric. A value between 0 and 1 indicating the proportion of cases and controls to use in analysis (if targetVector = "binary") or proportion of the full sample (if targetVector = "nonbinary"). Defaults to 0.8.
#'@param times Numeric. Indicates the number of replications to run with randomization.
#'@param targetVector Indicate "binary" for target vector with two options (e.g., case/control). Indicate "nonbinary" for target vector with real numbers.
#'@param optimize Criteria to optimize if targetVector = "binary." Indicate "pval" to optimize the p-value corresponding to the t-test distinguishing case and control. Indicate "auc" to optimize the AUC.  Defaults to pval.
#'@param outputPath The path where files are to be written as output, default is None meaning no files will be written.  When targetVector is "binary" file binary.csv will be output in the provided path, showing the reults.  When targetVector is "nonbinary" file nonbinary.csv will be output in the provided path, showing the results.  In the same path, the kept and unkept variables from the last iteration, will be output, prefixed with the targetVector type "binary" or "nonbinary" followed by Kept and Unkept and suffixed with .csv.  Two files containing the results from each run have List in the filenames and suffixed with .txt.
#'@return A data frame containing "times" rows of CALF runs where each row represents a run of CALF on a randomized "proportion" of "data".  Colunns start with the numer selected for the run, followed by AUC or pval and then all markers from "data".  An entry in a marker column signifys a chosen marker for a particular run (a row) and their assigned coarse weight (-1, 0, or 1).
#'@examples
#'\dontrun{
#'cv.calf(data = CaseControl, limit = 5, times = 100, targetVector = 'binary', optimize = 'pval')
#'}
#'@export
def calf_cv(data, targetVector, limit, times, proportion = .8, optimize = 'pval', outputPath=None):

	if targetVector != 'binary' and targetVector != 'nonbinary':
		raise Exception('CALF ERROR: Invalid targetVector argument.  Only "binary" or "nonbinary" is allowed.')
	elif targetVector == 'binary' and optimize=='corr':
		raise Exception('CALF ERROR: Optimizing by "corr" is only applicable to nonbinary data.')
	elif targetVector == 'nonbinary' and optimize=='pval':
		raise Exception('CALF ERROR: Optimizing by "pval" is only applicable to binary data.')
	elif targetVector == 'nonbinary' and optimize=='auc':
		raise Exception('CALF ERROR: Optimizing by "auc" is only applicable to binary data.')
	else:

		#Get the rows of interest first, as there is no reason to repeat this
		if targetVector == 'binary':

			ctrlRows = data.loc[data.iloc[:,0] == 0]
			caseRows = data.loc[data.iloc[:,0] == 1]

			# calculate number of case and control to keep
			nCtrlKeep = round(len(ctrlRows.index) * proportion)
			nCaseKeep = round(len(caseRows.index) * proportion)
			
		elif targetVector == 'nonbinary':
			nDataKeep = round(len(data)*proportion)


		header = None
		#Build the header row for the table that will be output
		columNames = data.columns.tolist()[1:]
		if targetVector == "binary":
			if optimize == 'pval':
				header = ["Number Selected", "AUC", "pval"] + columNames
			elif optimize == 'auc':
				header = ["Number Selected", "AUC"] + columNames
		elif targetVector == 'nonbinary':
			header = ["Number Selected", "corr"] + columNames


		results = pandas.DataFrame(0, index=range(0,times), dtype=object, columns=header)


		#Now run the CALF calculation "times" times
		rowCount = 0
		optimizedKeptList = list()
		optimizedUnkeptList = list()
		correlationList = list()
		
		while rowCount < times:

			if targetVector == 'binary':

				#Resample the binary data, keeping track of what was included and what was not.
				keepCtrlRows = ctrlRows.sample(n=nCtrlKeep)
				unkeptCtrlRows = ctrlRows.drop(keepCtrlRows.index.tolist())

				keepCaseRows = caseRows.sample(n=nCaseKeep)
				unkeptCaseRows = caseRows.drop(keepCaseRows.index.tolist())

				keepData = pandas.concat([keepCtrlRows, keepCaseRows])
				unkeptData = pandas.concat([unkeptCtrlRows, unkeptCaseRows])

				if outputPath is not None:
					outputFile = outputPath + 'binaryKept.csv'
					keepData.to_csv(outputFile, index=False)

					outputFile = outputPath + 'binaryExcluded.csv'
					unkeptData.to_csv(outputFile, index=False)

			elif targetVector == 'nonbinary':

				#Resample the nonbinary data
				keepData  = data.sample(n=nDataKeep)
				unkeptData = data.drop(keepData.index)

				if outputPath is not None:
					outputFile = outputPath + 'nonbinaryKept.csv'
					keepData.to_csv(outputFile, index=False)

					outputFile = outputPath + 'nonbinaryExcluded.csv'
					unkeptData.to_csv(outputFile, index=False)


			answer = calf_internal(data=keepData,
				nMarkers = limit,
				randomize  = False,
				proportion = None,
				times = 1,
				targetVector = targetVector,
				optimize = optimize,
				verbose = False)


			#Keep track of the optimizer values returned for each run
			if optimize == 'auc':
				results.at[rowCount,'AUC'] = answer['auc']
				optimizedKeptList.append(answer['auc'])
			elif optimize == 'pval':
				results.at[rowCount, "AUC"] = answer['auc']
				results.at[rowCount,"pval"] = answer['finalBest']
				
				optimizedKeptList.append(answer['finalBest'])
			elif optimize == 'corr':
				results.at[rowCount,'corr'] = answer['finalBest']
				optimizedKeptList.append(answer['finalBest'])
			
			#Keep a tally of the results per calf run
			markerList = answer['selection']['Marker']
			lenMarkerList = len(markerList)
			results.at[rowCount, "Number Selected"] = lenMarkerList

			markerCount = 0
			while markerCount < lenMarkerList:
				results.at[rowCount, markerList[markerCount]] = answer['selection']['Weight'][markerCount]
				markerCount += 1

			#Perform the cross-validation
			if targetVector == 'binary':
				if optimize == 'pval':
					unkeptDropped = unkeptData.drop(unkeptData.columns[0], axis=1)
					resultsDropped = results.drop(results.columns[0:3], axis=1)

					weightsTimesUnkept = unkeptDropped.dot(resultsDropped.iloc[rowCount,:])
					
					resultCtrlData = weightsTimesUnkept[unkeptCtrlRows.index]
					resultCaseData = weightsTimesUnkept[unkeptCaseRows.index]
					
					optimizedUnkeptList.append(ttest_ind(resultCaseData, resultCtrlData, equal_var=False).pvalue)

				elif optimize == 'auc':
					unkeptDropped = unkeptData.drop(unkeptData.columns[0], axis=1)
					resultsDropped = results.drop(results.columns[0:2], axis=1)

					weightsTimesUnkept = unkeptDropped.dot(resultsDropped.iloc[rowCount,:])
					
					resultCtrlData = weightsTimesUnkept[unkeptCtrlRows.index]
					resultCaseData = weightsTimesUnkept[unkeptCaseRows.index]

					optimizedUnkeptList.append(computeAuc(resultCaseData, resultCtrlData))

			elif targetVector == 'nonbinary':

				unkeptDropped = unkeptData.drop(unkeptData.columns[0], axis=1)
				resultsDropped = results.drop(results.columns[0:2], axis=1)
				weightsTimesUnkept = unkeptDropped.dot(resultsDropped.iloc[rowCount,:])
				corrResult = pandas.Series(weightsTimesUnkept, dtype=float).corr(pandas.Series(unkeptData.iloc[:,0]))

				correlationList.append(corrResult)
			
			
			#If an outputPath was provided, then output the extra data generated by the CV
			if outputPath is not None:
				#Write the results
				if targetVector == 'binary':
					outputFile = outputPath + 'binary.csv'
					results.to_csv(outputFile, index=False)

					outputFile = outputPath + optimize + 'KeptList.txt'
					optimizedKept = open(outputFile, "w")
					optimizedKept.write(str(optimizedKeptList))
					optimizedKept.close()

					outputFile = outputPath + 'AUCExcludedList.txt'
					optimizedUnkept = open(outputFile, "w")
					optimizedUnkept.write(str(optimizedUnkeptList))
					optimizedUnkept.close()

				elif targetVector == 'nonbinary':
					outputFile = outputPath + 'nonbinary.csv'
					results.to_csv(outputFile, index=False)

					outputFile = outputPath + 'corrExcludedList.txt'
					correlationExcluded = open(outputFile, "w")
					correlationExcluded.write(str(correlationList))
					correlationExcluded.close()

			rowCount += 1
			
		return(results)





#'@title write_calf
#'@description Writes data returned from a call to calf()
#'@param x A CALF randomize object returned from a calf() call.
#'@param filename The output filename
#'@export
def write_calf(x, filename):

	x['selection'].to_csv(filename, index = False, mode='w')

	file = open(filename,'a')
	file.write('\n')
	if x['finalBest'] is not None:
		if x['targetVec'] == 'binary' and x['optimize'] == 'auc':
			file.write('AUC,')
		elif x['targetVec'] == 'binary' and x['optimize'] == 'pval':
			file.write('pval,')
		elif x['targetVec'] == 'nonbinary':
			file.write('corr,')
	file.write(str(x['finalBest']))
	file.close()



#'@title write_calf_randomize
#'@description Writes data returned from a call to calf_randomize()
#'@param x A CALF randomize object returned from a calf_randomize() call.
#'@param filename The output filename
#'@export
def write_calf_randomize(x, filename):

	x['multiple'].to_csv(filename, index = False, mode='w')

	if x['times'] == 1:
		file = open(filename,'a')
		file.write('\n')
		if x['targetVec'] == 'binary' and x['optimize'] == 'auc':
			file.write('AUC,')
			file.write(str(x['auc']))
			file.close()
		elif x['targetVec'] == 'binary' and x['optimize'] == 'pval':
			file.write('pval,')
			file.write(str(x['finalBest']))
			file.close()
		elif x['targetVec'] == 'nonbinary':
			file.write('corr,')
			file.write(str(x['finalBest']))
			file.close()
	else:
		file = open(filename,'a')
		file.write('\n')
		file.close()
		if x['targetVec'] == 'binary' and x['optimize'] == 'auc':
			auc = pandas.DataFrame(x['auc'])
			auc.columns = ['AUC']
			auc.to_csv(filename, index = False, mode='a')
		elif x['targetVec'] == 'binary' and x['optimize'] == 'pval':
			finalBest = pandas.DataFrame(x['finalBest'])
			finalBest.columns = ['pval']
			finalBest.to_csv(filename, index = False, mode='a')
		elif x['targetVec'] == 'nonbinary':
			finalBest = pandas.DataFrame(x['finalBest'])
			finalBest.columns = ['corr']
			finalBest.to_csv(filename, index = False, mode='a')





#'@title write_calf_subset
#'@description Writes output of the CALF subset dataframe
#'@param x A CALF subset object returned from a calf_subset() call.
#'@param filename The output filename
#'@export
def write_calf_subset(x, filename):
	x['multiple'].to_csv(path_or_buf = filename,
		index = False,
		mode='w')

	file = open(filename,'a')
	file.write('\n')
	file.close()
	
 
	finalBest = pandas.DataFrame(x['finalBest'])
	if finalBest is not None:
		if x['targetVec'] == 'binary' and x['optimize'] == 'auc':
			finalBest.columns = ['AUC']
		elif x['targetVec'] == 'binary' and x['optimize'] == 'pval':
			finalBest.columns = ['pval']
		elif x['targetVec'] == 'nonbinary':
			finalBest.columns = ['corr']
		
		finalBest.to_csv(path_or_buf =  filename, index = False, mode = 'a')




def print_optimizer(result):
	if result['targetVec'] == 'binary':
		print('AUC: {}'.format(result['auc']))
		if result['optimize'] == 'pval':
			print('Final p-value: {}'.format(result['finalBest']))
	else:
		print('Final Correlation: {}'.format(result['finalBest']))
	
	

def calf_internal(
	data,
	nMarkers,
	randomize = False,
	proportion = None,
	times = 1,
	targetVector = 'binary',
	optimize = 'pval',
	verbose = False):

	x = None
	y = None
	refx = None
	refy = None

	if targetVector == 'nonbinary':
		optimize = None

	nVars = data.shape[1] - 1
	dNeg  = -data.iloc[:,1:data.shape[1]]
	dNeg.columns = list(map(lambda i: i+'.1',dNeg.columns))
	data = pandas.concat([data,dNeg], axis=1)

	if nMarkers > nVars:
		raise Exception('CALF ERROR: Requested number of markers is larger than the number of markers in data set.  Please revise this value or make sure your data were read in properly.')

	if randomize == True:
		data.iloc[:,0] = data.iloc[:,0].sample(frac=1).values


	if proportion is not None:
		if targetVector == 'binary':
			ctrlRows = data.loc[data.iloc[:,0] == 0]
			caseRows = data.loc[data.iloc[:,0] == 1]

			if type(proportion) is list:

				#When proportion is provided as a list of two values, the first for control and the second for case
				if len(proportion) == 2:
				
					#Check if each element of the list is numeric
					if all(isinstance(element, float) for element in proportion) == False:
						raise Exception('CALF ERROR: Proportion provided as list can only consist of numeric values.')
					if proportion[0] + proportion[1] != 1.0:
						raise Exception('CALF ERROR: The two values in the proportion list must sum to 1.0.')

					# Sample, randomly, rows of case and control to keep, record rows to keep
					ctrlRows = ctrlRows.sample(frac=proportion[0])
					caseRows = caseRows.sample(frac=proportion[1])
				else:
					raise Exception('CALF ERROR: Proportion provided as list value can only consist of two numbers.')

			else:
				# Sample, randomly, rows of case and control to keep, record rows to keep
				ctrlRows = ctrlRows.sample(frac=proportion)
				caseRows = caseRows.sample(frac=proportion)

			if len(ctrlRows) == 0 or len(caseRows) == 0:
				raise Exception('CALF ERROR: Proportion values provided result in either number of case or control equal to zero. Adjust your proportion and try again.')


			# subset original data to keep these rows
			data = pandas.concat([ctrlRows,caseRows])

		else:
			data = data.sample(frac=proportion)

	real = data.iloc[:,0]
	realMarkers = data.iloc[:,1:data.shape[1]]
	ctrl = data.loc[data.iloc[:,0] == 0].iloc[:,1:data.shape[1]]
	case = data.loc[data.iloc[:,0] == 1].iloc[:,1:data.shape[1]]
	indexNegPos = [0] * nVars * 2

	keepIndices = list()

	# initial loop to establish first optimal marker -------------------#
	allCrit = list()
	for i in range(0, nVars*2):
		if targetVector == 'binary':
			caseVar = case.iloc[:,i]
			ctrlVar = ctrl.iloc[:,i]
			if optimize == 'pval':
				crit = ttest_ind(caseVar, ctrlVar, equal_var=False).pvalue
			elif optimize == 'auc':
				crit = computeAuc(caseVar, ctrlVar)
				crit = 1/crit
		else:
			realVar = realMarkers.iloc[:,i]
			crit = pandas.Series(real).corr(pandas.Series(realVar))
			crit = 1/crit

		allCrit.append(crit)

	for i, n in enumerate(allCrit):
		if allCrit[i] < 0:
			allCrit[i] = float("Nan")


	# end of initial loop ----------------------------------------------#
	keepIndex = allCrit.index(numpy.nanmin(allCrit))
	keepMarker = realMarkers.columns[keepIndex]
	bestCrit = numpy.nanmin(allCrit)

	keepMarkers = list()
	keepMarkers.append(keepMarker)
	bestCrits = list()
	bestCrits.append(bestCrit)
	keepIndices.append(keepIndex)

	if verbose == True:
		if targetVector == "binary":
			if optimize == "pval":
				log.info('Selected: {} p-value = {}'.format(keepMarkers[-1],round(bestCrits[-1], 15)))
			elif optimize == "auc":
				log.info('Selected: {} AUC = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))
		elif targetVector == "nonbinary":
			log.info('Selected: {} Correlation = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))

	if nMarkers != 1:
		# second loop to add another marker --------------------------------#
		allCrit = list()
		realPrev = realMarkers.iloc[:,keepIndex]
		casePrev = case.iloc[:,keepIndex]
		ctrlPrev = ctrl.iloc[:,keepIndex]
		for i in range(0, nVars*2):
			if i != keepIndex:
				caseVar = casePrev + case.iloc[:,i]
				ctrlVar = ctrlPrev + ctrl.iloc[:,i]
				realVar = realPrev + realMarkers.iloc[:,i]
				if targetVector == "binary":
					if optimize == "pval":
						crit = ttest_ind(caseVar, ctrlVar, equal_var=False).pvalue
					elif optimize == "auc":
						crit = computeAuc(caseVar, ctrlVar)
						crit = 1/crit
				else:
					crit = pandas.Series(real).corr(pandas.Series(realVar))
					crit = 1/crit
			else:
				crit = float("NaN")
			allCrit.append(crit)
		# end of second loop ----------------------------------------------#

		for i, n in enumerate(allCrit):
			if allCrit[i] < 0:
				allCrit[i] = float("Nan")

		# check if the latest p is lower than the previous p
		proceed = True if bestCrit > numpy.nanmin(allCrit) else False

		if proceed:
			keepMarkers.append(realMarkers.columns[allCrit.index(numpy.nanmin(allCrit))])
			bestCrits.append(numpy.nanmin(allCrit))
			keepIndices.append(allCrit.index(numpy.nanmin(allCrit)))
			if len(keepMarkers) == nMarkers:
				proceed = False

		if verbose == True:
			if targetVector == "binary": 
				if optimize == "pval":
					log.info('Selected: {} p-value = {}'.format(keepMarkers[-1],round(bestCrits[-1], 15)))
				elif optimize == "auc":
					log.info('Selected: {} AUC = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))
			elif targetVector == "nonbinary":
				log.info('Selected: {} Correlation = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))

		# loop for third through nMarker ----------------------------------#
		while proceed == True:
			allCrit = list()
			casePrev = case.iloc[:,keepIndices].sum(axis=1)
			ctrlPrev = ctrl.iloc[:,keepIndices].sum(axis=1)
			realPrev = realMarkers.iloc[:,keepIndices].sum(axis=1)

			for i in range(0, nVars*2):
				if i not in keepIndices:
					caseVar = casePrev + case.iloc[:,i]
					ctrlVar = ctrlPrev + ctrl.iloc[:,i]
					realVar = realPrev + realMarkers.iloc[:,i]
					if targetVector == "binary":
						if optimize == "pval":
							crit = ttest_ind(caseVar, ctrlVar, equal_var=False).pvalue
						elif optimize == "auc":
							crit = computeAuc(caseVar, ctrlVar)
							crit = 1/crit
					else:
						crit = pandas.Series(real).corr(pandas.Series(realVar))
						crit = 1/crit
				else:
					crit = float("NaN")
				allCrit.append(crit)

			for i, n in enumerate(allCrit):
				if allCrit[i] < 0:
					allCrit[i] = float("Nan")

			proceed = True if bestCrits[-1] > numpy.nanmin(allCrit) else False

			if proceed == True:
				keepMarkers.append(realMarkers.columns[allCrit.index(numpy.nanmin(allCrit))])
				bestCrits.append(numpy.nanmin(allCrit))
				keepIndices.append(allCrit.index(numpy.nanmin(allCrit)))
				proceed	 = bestCrits[-1] < bestCrits[-2]
				
				if verbose == True:
					if targetVector == "binary":
						if optimize == "pval":
							log.info('Selected: {} p-value = {}'.format(keepMarkers[-1],round(bestCrits[-1], 15)))
						elif optimize == "auc":
							log.info('Selected: {} AUC = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))
					elif targetVector == "nonbinary":
						log.info('Selected: {} Correlation = {}'.format(keepMarkers[-1], round((1/bestCrits[-1]),15)))

			if len(keepMarkers) == nMarkers:
				proceed = False

	if verbose == True:
		print("\n")

	indexNegPos = numpy.array([0] * 2 * nVars)
	indexNegPos[[i for i in keepIndices if i > nVars]] = -1
	indexNegPos[[i for i in keepIndices if i <= nVars]] = 1

	# Produce the table of results
	output = pandas.DataFrame({'Marker': [i.replace('.1','') for i in keepMarkers], 'Weight': indexNegPos[keepIndices]})

	if targetVector == "nonbinary" or optimize == "auc":
		finalBestCrit = 1 / bestCrits[-1]
	else:
		finalBestCrit = bestCrits[-1]

	if targetVector == "binary":
		if nMarkers != 1 and len(keepIndices) != 1:
			funcValue = pandas.concat([case.iloc[:,keepIndices].sum(axis=1),ctrl.iloc[:,keepIndices].sum(axis=1)])
		else:
			funcValue = pandas.concat([case.iloc[:,keepIndices], ctrl.iloc[:,keepIndices]])
			
		funcValue = round(funcValue,8)
		# rank individual function values
		ranks = pandas.DataFrame.rank(pandas.DataFrame(funcValue))
		seqCaseCtrl = [1]*len(case) + [0] * len(ctrl)


		# set up plot -----------------------------------------------------#

		all_result = pandas.concat([funcValue, pandas.DataFrame(seqCaseCtrl).set_index(funcValue.index), ranks], axis= 1)
		all_result.columns = ['funcValue', 'seqCaseCtrl', 'ranks']
		all_result = all_result.sort_values(by='ranks')

		x_list = numpy.arange(0,1,1/(len(all_result)-1))
		x_list = numpy.append(x_list, 1)
		refx = pandas.DataFrame(x_list).set_index(all_result.index)

		y_list = numpy.arange(0,1,1/(len(all_result)-1))
		y_list = numpy.append(y_list, 1)
		refy = pandas.DataFrame(y_list).set_index(all_result.index)

		all_result = pandas.concat([all_result, refx, refy], axis = 1)

		initVal = all_result['seqCaseCtrl'].iloc[0]
		moveRight = len(case) if initVal == 0 else len(ctrl)
		moveUp = len(ctrl) if initVal == 0 else len(case)

		xs = pandas.Series([0]*len(all_result))
		ys = pandas.Series([0]*len(all_result))

		for i in range(1,len(all_result)):
			if all_result.iloc[i].loc['seqCaseCtrl'] == initVal:
				xs.iloc[i] = xs.at[i-1]
				ys.iloc[i] = ys.at[i-1] + 1/(moveUp-1)
			else:
				xs.iloc[i] = xs.iloc[i-1] + 1/(moveRight)
				ys.iloc[i] = ys.iloc[i-1]
			
		all_result = pandas.concat([all_result, pandas.DataFrame(xs).set_index(all_result.index), pandas.DataFrame(ys).set_index(all_result.index)], axis= 1)
		all_result.columns = ['funcValue', 'seqCaseCtrl', 'ranks','refx', 'refy', 'x', 'y']


		# if the plot prints upside-down, switch values for
		# x and y
		n = round(len(ys)/2, 0)
		rocPlot = ggplot(all_result) + \
		geom_line(mapping=aes(x = 'x', y = 'y'), color='black', size=1) + \
		geom_line(mapping=aes(x = 'refx', y ='refy'), color='red', size=1.5) +\
		scale_x_continuous(limits = (0,1)) +\
		theme_bw() + \
		theme(legend_position = None) + \
		labs(x='False Positive Rate (1 - Specificity)', y='True Positive Rate (Sensitivity)')
		#rocPlot.save('filename.pdf', height=6, width=8)
		
		# set up plot -----------------------------------------------------#

		# compute arguments for AUC
		caseFunc = ranks.iloc[0:len(case)].sum().sum() - len(case)*(len(caseVar)+1)/2
		ctrlFunc = ranks.iloc[(len(case)):len(ranks)].sum().sum() - len(ctrl)*(len(ctrl)+1)/2
		auc = round(max(ctrlFunc, caseFunc)/(caseFunc + ctrlFunc),4)

	else:
		auc = None
		rocPlot = None


	return {'selection': output,
			'auc': auc,
			'randomize': randomize,
			'proportion': proportion,
			'targetVec': targetVector,
			'rocPlot': rocPlot,
			'finalBest': finalBestCrit,
			'optimize': optimize}
