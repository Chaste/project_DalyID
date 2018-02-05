# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Experiment imports
from modeling.simulation.fcexperiment import FunctionalCurationExperiment

# Fitting imports
import modeling.language.distributions as Dist
from modeling.language.kernels import Kernel
from modeling.fitting.objective import SquareError, LogLikGauss, MeanEuclidean
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.approximatebayes import ABCSMCDelMoral
from modeling.fitting.MCMC import AdaptiveCovarianceMCMC

# General imports
import numpy, csv
from modeling.utility.io import ReadParameterDistribution, WriteDataSet, ReadDataSet
from modeling.utility.plotting import PlotDiscreteMarginals, PlotHeatMap
from modeling.utility.translate import Rename

import time

class HodgkinHuxleyFittingSimon(unittest.TestCase):

	'''	Fit the Hodgkin-Huxley model to summary statistics calculated from noisy action
		potentials (provided in test/data/HHSumStatsFromNoisyAP.dat).
	'''
	def TestNoisyFit(self):
		# Model parameters 
		trueParameters = {}
		trueParameters['oxmeta:membrane_fast_sodium_current_conductance'] = 120
		trueParameters['oxmeta:membrane_potassium_current_conductance'] = 36
		trueParameters['aidan:leakage_current_max'] = 0.3

		# Shortened names for plotting
		shortNames = {}
		shortNames['oxmeta:membrane_fast_sodium_current_conductance'] = 'G_Na'
		shortNames['oxmeta:membrane_potassium_current_conductance'] = 'G_K'
		shortNames['aidan:leakage_current_max'] = 'G_l'

		modelFile = 'projects/DalyID/hodgkin_huxley.cellml'
		protoFile = 'projects/DalyID/test/protocols/hh_aptrace_sumstats.txt'
		experiment = FunctionalCurationExperiment(protoFile,modelFile)

		# Define uniform priors over model parameters 
		priors = {}
		for oxmeta,val in trueParameters.iteritems():
			priors[oxmeta] = Dist.Uniform(val*0.5,val*2.)
		priorDist = Dist.IndependentParameterDistribution(priors)
		ranges = {'G_Na':[60,240],'G_K':[18,72],'G_l':[.15,.6]}

		# Calculate summary statistics under reported parameters
		# (Used to scale the components of the distance function)
		sumstats = ['APD90','APD50','PeakPotential','RestPotential','MaxUpstrokeVel']
		results = experiment.simulate()
		simData = dict([(ss,results[ss]) for ss in sumstats])
		expData = ReadDataSet("projects/AidanDaly/test/data/HHSumStatsFromNoisyAP.dat")

		# Distance between data simulated under reported parameters and experimental data
		objFun = MeanEuclidean()
		objArgs = {'std':"exp"} # Normalize relative to experimental data 

		# Define the summary statistics to use
		ss = ['PeakPotential','RestPotential','APD50']
		outputMapping = dict([(name,name) for name in ss])
		fname = "HHSumStats-ABC/ABC_"+("-".join(ss))

		# Define the fitting algorithm
		algorithm = ABCSMCDelMoral()
		algArgs = {'cutoff':0.001,'postSize':100000,'alpha':0.2,'minESS':0.3,
			'resampleESS':0.6,'outputFile':fname+".out"}
		algArgs['outputFile'] = "HHSumStats-ABC/ABC_"+("-".join(outputMapping.keys()))+".out"

		task = ParameterFittingTask(priorDist,experiment,expData,objFun,
			outputMapping=outputMapping,objArgs=objArgs)

		# Define start/end conditions for sampler
		minErr = 0.075
		algArgs['minErr'] = minErr
		algArgs['e0'] = 10*minErr

		# Run ABC sampler and plot results
		results = algorithm(task,args=algArgs)
		dist = Rename(results,shortNames)
		PlotHeatMap(dist,fname,plotRange=ranges,separateFigs=False,
			include=["G_Na","G_K","G_l"],fontsize=20,bins=100)

	def TestSingles(self):
		algorithm = AdaptiveCovarianceMCMC()
		algArgs = {'numIters':1e6,'burn':1e5}
		objFun = LogLikGauss()

		ss_names = ['PeakPotential','RestPotential','APD90','APD50','MaxUpstrokeVel']

		singles = []
		singles.append(['PeakPotential'])
		singles.append(['APD90'])
		singles.append(['APD50'])
		singles.append(['RestPotential'])
		singles.append(['MaxUpstrokeVel'])

		for i in range(0,len(singles)):
			sumstats = singles[i]
			baseName = 'HHSumStats-12-11/'+'-'.join(sumstats)

			self.FitToSelectedStats(sumstats,algorithm,algArgs,objFun,baseName)

	def TestPairs(self):
		algorithm = AdaptiveCovarianceMCMC()
		algArgs = {'numIters':1e6,'burn':1e5}
		objFun = LogLikGauss()

		ss_names = ['PeakPotential','RestPotential','APD90','APD50','MaxUpstrokeVel']

		pairs = []
		pairs.append(['PeakPotential','RestPotential'])
		pairs.append(['PeakPotential','APD90'])
		pairs.append(['PeakPotential','APD50'])
		pairs.append(['PeakPotential','MaxUpstrokeVel'])
		pairs.append(['RestPotential','APD90'])
		pairs.append(['RestPotential','APD50'])
		pairs.append(['RestPotential','MaxUpstrokeVel'])
		pairs.append(['APD90','APD50'])
		pairs.append(['APD90','MaxUpstrokeVel'])
		pairs.append(['APD50','MaxUpstrokeVel'])

		for i in range(0,len(pairs)):
			sumstats = pairs[i]
			baseName = 'HHSumStats-12-11/'+'-'.join(sumstats)

			self.FitToSelectedStats(sumstats,algorithm,algArgs,objFun,baseName)

	def TestTriples(self):
		algorithm = AdaptiveCovarianceMCMC()
		algArgs = {'numIters':1e6,'burn':1e5}
		objFun = LogLikGauss()

		ss_names = ['PeakPotential','RestPotential','APD90','APD50','MaxUpstrokeVel']

		triples = []
		triples.append(['PeakPotential','RestPotential','APD90'])
		triples.append(['PeakPotential','RestPotential','APD50'])
		triples.append(['PeakPotential','RestPotential','MaxUpstrokeVel'])
		triples.append(['PeakPotential','APD90','APD50'])
		triples.append(['PeakPotential','APD90','MaxUpstrokeVel'])
		triples.append(['PeakPotential','APD50','MaxUpstrokeVel'])
		triples.append(['RestPotential','APD90','APD50'])
		triples.append(['RestPotential','APD90','MaxUpstrokeVel'])
		triples.append(['RestPotential','APD50','MaxUpstrokeVel'])
		triples.append(['APD90','APD50','MaxUpstrokeVel'])

		for i in range(0,len(triples)):
			sumstats = triples[i]
			baseName = 'HHSumStats-12-11/'+'-'.join(sumstats)

			self.FitToSelectedStats(sumstats,algorithm,algArgs,objFun,baseName)

	'''	Fit the Hodgkin-Huxley model to specified summary statistics using MCMC
	'''
	def FitToSelectedStats(self,sumstats,algorithm,algArgs,objFun,baseName):
		# Model parameters 
		trueParameters = {}
		trueParameters['oxmeta:membrane_fast_sodium_current_conductance'] = 120
		trueParameters['oxmeta:membrane_potassium_current_conductance'] = 36
		trueParameters['aidan:leakage_current_max'] = 0.3

		# Shortened names for plotting
		shortNames = {}
		shortNames['oxmeta:membrane_fast_sodium_current_conductance'] = 'G_Na'
		shortNames['oxmeta:membrane_potassium_current_conductance'] = 'G_K'
		shortNames['aidan:leakage_current_max'] = 'G_l'

		# Specify the experiment
		modelFile = 'projects/AidanDaly/hodgkin_huxley.cellml'
		if "APD90" in sumstats or "APD50" in sumstats:
			protoFile = 'projects/DalyID/test/protocols/hh_aptrace_sumstats.txt'
		else:
			print "=== No APD calulation necessary ==="
			protoFile = 'projects/DalyID/test/protocols/hh_aptrace_sumstats_noapd.txt'
		experiment = FunctionalCurationExperiment(protoFile,modelFile)

		# Define uniform priors over model parameters 
		priors = {}
		for oxmeta,val in trueParameters.iteritems():
			priors[oxmeta] = Dist.Uniform(val*0.5,val*2.)
		priorDist = Dist.IndependentParameterDistribution(priors)

		# Gaussian proposal distributions with width 1/10 of the prior
		kerns = {}
		for oxmeta,val in trueParameters.iteritems():
			kerns[oxmeta] = Dist.Normal(0.,val*0.15)
		propDist = Kernel(Dist.IndependentParameterDistribution(kerns))
		algArgs['proposalDist'] = propDist
		algArgs['start'] = trueParameters

		# Set up the parameter fitting parameter fitting task
		# Define noise as a FRACTION of default value for each parameter INDIVIDUALLY with
		#  sigma_e, then generate noisy data.
		sigma_e = 0.01
		sigmas = {}
		expData = experiment.simulate(trueParameters)
		for ss in sumstats:
			print ss, expData[ss]
			expData[ss] += numpy.random.normal(scale=sigma_e * numpy.abs(expData[ss]))
			sigmas[ss] = sigma_e * numpy.abs(expData[ss])
			print ss, expData[ss]
		objArgs = {'std':sigmas}

		# Set up the MCMC fitting
		mapping = dict([(ss,ss) for ss in sumstats])
		task = ParameterFittingTask(priorDist,experiment,expData,objFun,
			outputMapping=mapping,objArgs=objArgs)
		algArgs['outputFile'] = baseName+'.out'
		algArgs['minErr'] = sigma_e
		
		# Run fitting and plot results
		results = algorithm(task,args=algArgs)
		results = Rename(results,shortNames)

		ranges = {'G_Na':[60,240],'G_K':[18,72],'G_l':[.15,.6]}
		PlotHeatMap(results,baseName,plotRange=ranges,separateFigs=False,
			include=["G_Na","G_K","G_l"],fontsize=20,bins=100)


