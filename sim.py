import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

N_arms = 10
N_users = 5
ucb_alpha = 1
t = 1
actualBernMeans = np.random.uniform(0,1,(N_users,N_arms))

class user :
	def __init__(self,nbu):
		self.nbu = nbu
		self.alpha = np.ones(N_arms)
		self.beta = np.ones(N_arms)

		self.est_mean = 0
		self.ucb = 0
		self.num_pulled = np.zeros(N_arms)

		self.regret = 0

	def updateThompDistr(self,arm,bern):
		self.alpha[arm] = self.alpha[arm] + bern
		self.beta[arm] = self.beta[arm] + 1 - bern				

	def updateUCBEst(self,arm,bern):
		num_pulled[arm] += 1
		self.est_mean = (self.est_mean*(num_pulled[arm] - 1) + bern)/(num_pulled[arm])
		self.ucb = math.sqrt(ucb_alpha*math.log(t) / (2*num_pulled[arm]))

def updateThompGraph(users,arms,armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived])-actualBernMeans[userArrived][armPulled]
	userArrived.regret += inst_regret

	userArrived.updateThompDistr(armPulled,reward)

	for i in range(userArrived.nbu.length()):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateThompDistr(armPulled,reward1)

	for i in range(armGraph[armPulled].length()):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived][armGraph[armPulled][i]])
		userArrived.updateThompDistr(armGraph[armPulled][i],reward2)

def updateUCBGraph(users,arms,armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived])-actualBernMeans[userArrived][armPulled]
	userArrived.regret += inst_regret

	userArrived.updateUCBEst(armPulled,reward)

	for i in range(userArrived.nbu.length()):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateUCBEst(armPulled,reward1)

	for i in range(armGraph[armPulled].length()):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived][armGraph[armPulled][i]])
		userArrived.updateUCBEst(armGraph[armPulled][i],reward2)



