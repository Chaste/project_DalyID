# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Experiment imports
from modeling.simulation.LogisticGrowthExperiment import LogisticExperiment

# Fitting imports
import modeling.language.distributions as Dist
from modeling.language.kernels import Kernel
from modeling.fitting.objective import LogLikGauss
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.MCMC import MetropolisHastingsMCMC

# General imports
import numpy
from modeling.utility.io import ReadParameterDistribution
from modeling.utility.plotting import PlotDiscreteMarginals, PlotHeatMap

class LogisticFittingSimon(unittest.TestCase):
	
	def RunMCMC(self,times,outputName):
		# Model parameters
		K = 17.5 		# The carrying capacity of the model
		r = 0.7			# The growth rate of the model
		x0 = 0.1 		# The (known) initial condition of the model
		sigma_e = 1.0 	# The (known) standard deviation of the observation error

		# Specify the experiment
		experiment = LogisticExperiment(times,x0=x0)

		# Define true parameters, true solution, and generate noisy observation
		theta_0 = {'K':K,'r':r}
		trueSolution = experiment.simulate(theta_0)

		# Define uniform priors over model parameters
		prior_K = Dist.Uniform(x0,200.)
		prior_r = Dist.Uniform(0.,10.)
		priorDist = Dist.IndependentParameterDistribution({'K':prior_K,'r':prior_r})

		# Set up the parameter fitting task
		noisyData = {'x':[max(p+numpy.random.normal(scale=sigma_e),0) for p in trueSolution['x']]}
		objFun = LogLikGauss()
		objArgs = {'std':sigma_e}
		task = ParameterFittingTask(priorDist,experiment,noisyData,objFun,objArgs=objArgs)

		# Set up MCMC fitting
		algorithm = MetropolisHastingsMCMC()
		kernDists = {'K':Dist.Normal(0,(200.-x0)/10),'r':Dist.Normal(0,1.)}
		propDist = Kernel(Dist.IndependentParameterDistribution(kernDists))
		algArgs = {'proposalDist':propDist,'tune':True,'numIters':6e4,'burn':3e4,
			'outputFile':outputName+'.out'}
		
		# Run fitting and plot output
		results = algorithm(task,algArgs)
		PlotHeatMap(results,outputName)

	def TestStart(self):
		times = [1.,2.,3.]
		outputName = 'SimonLogistic/LogisticSimonMCMC_tstart'
		self.RunMCMC(times,outputName)

	def TestMid(self):
		times = [6.,7.,8.]
		outputName = 'SimonLogistic/LogisticSimonMCMC_tmid'
		self.RunMCMC(times,outputName)

	def TestEnd(self):
		times = [22.,23.,24.]
		outputName = 'SimonLogistic/LogisticSimonMCMC_tend'
		self.RunMCMC(times,outputName)

	def TestMixed(self):
		times = [2.,7.,24.]
		outputName = 'SimonLogistic/LogisticSimonMCMC_tmixed'
		self.RunMCMC(times,outputName)

