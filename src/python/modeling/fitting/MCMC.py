'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

import algorithm as Algorithm
from ..language import distributions as Dist
from ..language import kernels as Kern
from ..utility.io import WriteParameterDistribution

import numpy
import sys, traceback

class MetropolisHastingsMCMC(Algorithm.ParameterFittingAlgorithm):
	@classmethod
	def __call__(cls,task,args={}):

		numIters = 10000
		burn = 5000
		start = task.prior.draw()

		tune = True
		tuneInterval = 1000
		tuneThroughout = True

		outputFile = None

		if args == None:
			args = {}
		if 'numIters' in args:
			numIters = args['numIters']
		if 'burn' in args:
			burn = min(args['burn'],numIters)
		if 'start' in args:
			start = args['start']
			assert task.prior.pdf(start) > 0, "Illegal starting point (0 prior likelihood)"
		if 'tune' in args:
			tune = args['tune']
		if 'tuneInterval' in args:
			tuneInterval = args['tuneInterval']
		if 'tuneThroughout' in args:
			tuneThroughout = args['tuneThroughout']
		if 'outputFile' in args:
			outputFile = args['outputFile']

		# Default to Gaussian proposal distribution in absence of proposal kernel
		# 'proposalDist' may be provided as a Kernel or a ParameterDistribution
		if not 'proposalDist' in args:
			print "Generating Gaussian kernel from prior samples..."
			kern = Kern.GenerateGaussianKernel(task.prior)
		else:
			# TODO: assert distribution matches prior?
			if isinstance(args['proposalDist'],Kern.Kernel):
				kern = args['proposalDist']
			else:
				kern = Kern.Kernel(args['proposalDist'])

		numAccepted = 0
		numRejected = 0
		iters = 1
		accepted = []
		current = start

		#print '\t'.join([str(val) for key,val in current.iteritems()])

		while iters <= numIters:
			# Tuning performed on this step
			if tune and iters % tuneInterval == 0 and (tuneThroughout or iters < burn):
				acceptanceRate = float(numAccepted)/(numAccepted+numRejected)
				print "Tuning: acceptance rate = "+str(acceptanceRate)
				
				cls.tuneStep(kern,acceptanceRate)
				numAccepted = 0
				numRejected = 0

			proposed = cls.proposeStep(task,kern,current,accepted)

			if cls.acceptStep(task,kern,current,proposed):
				#print proposed
				current = proposed
				numAccepted += 1
				#print '\t'.join([str(val) for key,val in current.iteritems()])
			else:
				numRejected += 1
			if iters > burn:
				accepted.append(current)

			iters += 1

			if iters == burn:
				print "Burn-in complete"

		if outputFile != None:
			cls.writeChain(accepted,outputFile)
		
		post = Dist.DiscreteParameterDistribution(accepted)
		return post

	''' 
		Tuning strategy adopted by the PyMC implementation of Metropolis-Hastings MCMC
		Returns False if tuning was not needed on this step, True otherwise 
	'''
	@classmethod
	def tuneStep(cls,proposalDist,acceptanceRate):
		if not hasattr(proposalDist,'scale'):
			return False

		scale = proposalDist.scale		

		if acceptanceRate < 0.001:
			proposalDist.scale *= 0.1
		elif acceptanceRate < 0.05:
			proposalDist.scale *= 0.5
		elif acceptanceRate < 0.2:
			proposalDist.scale *= 0.9
		elif acceptanceRate > 0.95:
			proposalDist.scale *= 10.0
		elif acceptanceRate > 0.75:
			proposalDist.scale *= 2.0
		elif acceptanceRate > 0.5:
			proposalDist.scale *= 1.1
		else:
			return False

		# Prevent from tuning to 0
		if proposalDist.scale == 0:
			proposalDist.scale = scale

		return True

	'''
		Allow posteriorEstimate as an argument in case step width is to be set based
		on current variability of parameters.
	'''
	@classmethod
	def proposeStep(cls,task,kern,current,posteriorEstimate):
		return kern.perturb(current)
		
	'''
		Metropolis-Hastings acceptance criteria, with additional check to prior 
	'''
	@classmethod
	def acceptStep(cls,task,kern,current,proposed):
		# Discount impossible proposals
		if task.prior.pdf(proposed) == 0:
			return False

		# If the prior is uniform and the proposal distribution is symmetric, these will be 0
		hastingsFactor = numpy.log(kern(proposed,current)) - numpy.log(kern(current,proposed))
		logPriorRatio = numpy.log(task.prior.pdf(proposed)) - numpy.log(task.prior.pdf(current))

		logLikRatio = task.calculateObjective(proposed) - task.calculateObjective(current)

		# Toss the die
		r = numpy.log(numpy.random.rand())
		return (r <= (logLikRatio + logPriorRatio + hastingsFactor))

	# TODO: Make use of io.WriteParameterDistribution()
	@classmethod
	def writeChain(cls,accepted,outputFile):
		outputFileHandle = open(outputFile,'w')
		keys = accepted[0].keys()
		
		# Write the header
		outputFileHandle.write("\t".join(keys))
		outputFileHandle.write("\n")

		# Write the progress of the chain from top to bottom
		for step in accepted:
			outputFileHandle.write("\t".join([str(step[k]) for k in keys]))
			outputFileHandle.write("\n")



class AdaptiveCovarianceMCMC(MetropolisHastingsMCMC):
	@classmethod
	def __call__(cls,task,args={}):

		numIters = 10000
		burn = numIters/2
		start = task.prior.draw()
		tuneThroughout = True

		outputFile = None

		if args == None:
			args = {}
		if 'numIters' in args:
			numIters = args['numIters']
		if 'burn' in args:
			burn = min(args['burn'],numIters-1)
		else:
			burn = int(numIters/2)
		if 'start' in args:
			start = args['start']
			assert task.prior.pdf(start) > 0, "Illegal starting point (0 prior likelihood)"
		else:
			print "NO STARTING POINT SPECIFIED - Choosing randomly from prior"
		if 'outputFile' in args:
			outputFile = args['outputFile']
		if 'tuneThroughout' in args:
			tuneThroughout = args['tuneThroughout']

		# Default to Gaussian proposal distribution in absence of proposal kernel
		# 'proposalDist' may be provided as a Kernel or a ParameterDistribution
		if not 'proposalDist' in args:
			print "Generating Gaussian kernel from prior samples..."
			kern = Kern.GenerateGaussianKernel(task.prior)
		else:
			# TODO: assert distribution matches prior?
			if isinstance(args['proposalDist'],Kern.Kernel):
				assert hasattr(args['proposalDist'].parameterDistribution,'std') or isinstance(args['proposalDist'].parameterDistribution,Dist.MVNParameterDistribution)
				kern = args['proposalDist']
			else:
				assert hasattr(args['proposalDist'],'std') or isinstance(args['proposalDist'],Dist.MVNParameterDistribution)
				kern = Kern.Kernel(args['proposalDist'])

		# Parameters for tuning proposal distribution
		a = 1.0

		# Transform kernel into MVN form
		if not hasattr(kern.parameterDistribution,'cov'):
			std = kern.parameterDistribution.std()
			pnames = std.keys()

			meanDict = kern.parameterDistribution.mean()
			meanVec = [meanDict[p] for p in pnames]
			Sigma = a*numpy.diag([std[p]**2 for p in pnames])

			kern = Kern.Kernel(Dist.MVNParameterDistribution(pnames,meanVec,Sigma))
		# Currently assumes MVN form
		else:
			Sigma = kern.parameterDistribution.cov()
		
		mu = numpy.array([start[p] for p in kern.parameterDistribution.pnames])

		numAccepted = 0
		numRejected = 0
		iters = 0
		accepted = []
		current = start

		while iters <= numIters:
			# Dealing with a weird error where "draw" fails due to "ValueError: array must not contain infs or NaNs"
			try:
				proposed = cls.proposeStep(task,kern,current,accepted)
				accept, paccept = cls.acceptStep(task,kern,current,proposed)

				if accept:
					current = proposed
					numAccepted += 1
				else:
					numRejected += 1
				if iters > burn:
					accepted.append(current)
				acceptanceRate = float(numAccepted)/(numAccepted+numRejected)

				iters += 1

				if iters == burn:
					print "Burn-in complete"

				# Tuning performed on this step
				if tuneThroughout or iters < burn:

					gamma = (iters+1)**-0.7
					currVec = numpy.array([current[p] for p in kern.parameterDistribution.pnames])
					mu, Sigma, a = cls.tuneStep(kern,currVec,paccept,gamma,mu,Sigma,a)

					#print "ITER: "+str(iters)+" - Accept: "+str(accept)+" ("+str(acceptanceRate)+")"
					#print Sigma

				# Print progress
				pctComplete = 100*float(iters)/numIters
				if pctComplete % 1 == 0:
					print str(pctComplete)+"% complete ("+str(iters)+" steps)"
					print "Proposal distribution:"
					print kern.parameterDistribution.covMat
					print "Acceptance rate: " + str(acceptanceRate)

					if iters > burn:
						tmp = Dist.DiscreteParameterDistribution(accepted)
						print "Post. Mean: ", tmp.mean()

						if outputFile != None and iters > burn:
							cls.writeChain(accepted,outputFile)
			
			except ValueError:
				print iters, mu, Sigma, a, start
				print gamma, accept, paccept
				break


		if outputFile != None:
			cls.writeChain(accepted,outputFile)
		
		post = Dist.DiscreteParameterDistribution(accepted)
		return post

	''' 
		Tuning strategy proposed by Remi and implemented by Dave
		Adaptively scales covariance matrix based on acceptance rate with a growing
			damping factor as the iterations proceed
	'''
	@classmethod
	def tuneStep(cls,kern,curr,paccept,gamma,mu,Sigma,a):
		delta = 1e-15

		new_mu = mu + gamma * (curr - mu)

		new_Sigma = Sigma + gamma * (numpy.outer(curr-mu,curr-mu) - Sigma) 
		
		# Commenting this out to see if it is the reason that my MCMC chains are not matching Kylie's in the long term
		# Perhaps this should be replaced with a check that re-sets Sigma in the event it becomes singular? But perhaps
		#   checking the determinant each iteration would be costly...
		#new_Sigma += delta*numpy.identity(len(mu)) # To combat singularity

		if numpy.isnan(numpy.log(a)):
			print "\t\tERROR: a = "+a
			a = 1.0 

		log_a = numpy.log(a) + gamma * (paccept - 0.234)
		new_a = numpy.exp(log_a)

		kern.parameterDistribution.covMat = new_a*new_Sigma
		return new_mu, new_Sigma, new_a

	'''
		Metropolis-Hastings acceptance criteria, with additional check to prior 
	'''
	@classmethod
	def acceptStep(cls,task,kern,current,proposed):
		# Discount impossible proposals
		if task.prior.pdf(proposed) == 0:
			return False, 0

		logLikRatio = task.calculateObjective(proposed) - task.calculateObjective(current)

		# Toss the die
		r = numpy.log(numpy.random.rand())
		return (r <= (logLikRatio)), min(numpy.exp(logLikRatio),1)
