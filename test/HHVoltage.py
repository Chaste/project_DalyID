'''
	These tests were written to determine whether choosing points to observe the 
	membrane voltage based on points of maximal leading eigenvalues in the FIM 
	will constrain the model better than naive uniform observation

	Three schemes to test:
	- Choose 3 points uniformly across AP
	- Choose 3 points of maximal lambda_1: 
	- Choose 3 points of minimal lambda_1: 
	Perform MCMC using these measurements

'''

# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy

# Experiment imports
from modeling.simulation.fcexperiment import FunctionalCurationExperiment

# Distributional imports
import modeling.language.distributions as Dist

# Fitting imports
from modeling.fitting.objective import LogLikGauss
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.MCMC import AdaptiveCovarianceMCMC

# I/O and plotting imports
from modeling.utility.plotting import PlotHeatMap
from modeling.utility.translate import Rename
from modeling.utility.io import ReadParameterDistribution

class OptimalObservation(unittest.TestCase):

	def TestUniform(self):
		times = [3.,6.,9.]
		baseName = "HHOptimalTimes/Uniform"

		self.RunAtTimes(times,baseName)

	def TestMaxEig(self):
		times = [1.2,2.3,7.2]
		baseName = "HHOptimalTimes/MaxEig"

		self.RunAtTimes(times,baseName)

	def TestMinEig(self):
		times = [1.5,4.0,11.4]
		baseName = "HHOptimalTimes/MinEig"

		self.RunAtTimes(times,baseName)

	def RunAtTimes(self,times,baseName):
		# Model parameters 
		trueParameters = {}
		trueParameters['oxmeta:membrane_fast_sodium_current_conductance'] = 120
		trueParameters['oxmeta:membrane_potassium_current_conductance'] = 36
		trueParameters['aidan:leakage_current_max'] = 0.3
		sigma_e = 0.2

		# Shortened names for plotting
		shortNames = {}
		shortNames['oxmeta:membrane_fast_sodium_current_conductance'] = 'G_Na'
		shortNames['oxmeta:membrane_potassium_current_conductance'] = 'G_K'
		shortNames['aidan:leakage_current_max'] = 'G_l'

		# Define uniform priors over model parameters 
		priors = {}
		for oxmeta,val in trueParameters.iteritems():
			priors[oxmeta] = Dist.Uniform(val*0.5,val*2.)
		priorDist = Dist.IndependentParameterDistribution(priors)

		# Define the experiment at the desired time points
		modelfile = "projects/DalyID/hodgkin_huxley.cellml"
		protofile = "projects/DalyID/test/protocols/hh_aptrace_fixedpoints.txt"

		exp = FunctionalCurationExperiment(protofile, modelfile)
		exp.setInputs({'sim_times':times})

		# Generate noisy data
		results = exp.simulate()
		noisyData = {'V':[v+numpy.random.normal(scale=sigma_e) for v in results['V']]}
		objArgs = {'std':sigma_e}

		# Perform MCMC fitting
		mapping = {'V':'V'}
		objFun = LogLikGauss()
		task = ParameterFittingTask(priorDist,exp,noisyData,objFun,
			outputMapping=mapping,objArgs=objArgs)
		algArgs = {'numIters':4e4,'burn':3e4,'outputFile':baseName+".out"}
		
		mcmc = AdaptiveCovarianceMCMC()
		results = mcmc(task,args=algArgs)
		results = Rename(results,shortNames)

		ranges = {'G_Na':[60,240],'G_K':[18,72],'G_l':[.15,.6]}
		PlotHeatMap(results,baseName,plotRange=ranges,separateFigs=False,
			include=["G_Na","G_K","G_l"],fontsize=20)
		
