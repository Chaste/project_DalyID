# Experiment class for standard time-course observation of the Velhurst-Pearl
# logistic population growth model:
#
#   dx/dt = rx(1-x/K)
#   
# Which has the analytic solution:
#
#   x(t) = K/(1 + (K/x_0 - 1)e^(-rt))
#
# The model contains three parameters:
#   K - carrying capacity (asymptotic value)
#   r - growth rate
#   x_0 - initial population size

from experiment import Experiment
import numpy


class LogisticExperiment(Experiment):
	def __init__(self,times,x0=1):
		# Time range - 0:0.1:25
		self.times = times
		self.x0 = x0

	def simulate(self,parameters):
		xs = numpy.array([self._analytic(parameters,t) for t in self.times])
		return {'x':xs}

	def _analytic(self,parameters,t):
		K = float(parameters['K'])
		r = parameters['r']

		return K/(1+(K/self.x0-1)*numpy.exp(-r*t))

	def jacobian(self,parameters):
		jac = numpy.zeros((len(self.times),2))
		for i,t in enumerate(self.times):
			fact = ( 1 + (parameters['K']/self.x0 - 1) * numpy.exp(-parameters['r']*t) )**(-2)
			jac[i][0] = fact * ((1 + (parameters['K']/self.x0 - 1)*numpy.exp(-parameters['r']*t)) - parameters['K']*numpy.exp(-parameters['r']*t)/self.x0)
			jac[i][1] = fact * (parameters['K'] * t * (parameters['K']/self.x0 - 1) * numpy.exp(-parameters['r']*t))
		return jac