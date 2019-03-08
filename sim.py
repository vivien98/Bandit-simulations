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
		self.nbu = nbu 						#neighbouring users in graph

		self.alpha = np.ones(N_arms)    #beta params
		self.beta = np.ones(N_arms)

		self.est_mean = np.zeros(N_arms)      #ucb algorithm params
		self.ucb = np.zeros(N_arms)
		self.num_pulled = np.zeros(N_arms)

		self.regret = 0     # regret

	def updateThompDistr(self,arm,bern):
		self.alpha[arm] = self.alpha[arm] + bern
		self.beta[arm] = self.beta[arm] + 1 - bern				

	def updateUCBEst(self,arm,bern):
		num_pulled[arm] += 1
		self.est_mean[arm] = (self.est_mean[arm]*(self.num_pulled[arm] - 1) + bern)/(self.num_pulled[arm])
		self.ucb[arm] = math.sqrt(ucb_alpha*math.log(t) / (2*self.num_pulled[arm]))

	def updateUCBTime(self):
		for i in range(N_arms):
			self.ucb[i] = math.sqrt(ucb_alpha*math.log(t) / (2*self.num_pulled[i]))



def updateThompGraph(armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived])-actualBernMeans[userArrived][armPulled]
	userArrived.regret += inst_regret

	userArrived.updateThompDistr(armPulled,reward)

	for i in range(userArrived.nbu.size):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateThompDistr(armPulled,reward1)

	for i in range(armGraph[armPulled].size):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived][armGraph[armPulled][i]])
		userArrived.updateThompDistr(armGraph[armPulled][i],reward2)

def updateUCBGraph(armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived])-actualBernMeans[userArrived][armPulled]
	userArrived.regret += inst_regret

	userArrived.updateUCBEst(armPulled,reward)

	for i in range(userArrived.nbu.size):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateUCBEst(armPulled,reward1)

	for i in range(armGraph[armPulled].size):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived][armGraph[armPulled][i]])
		userArrived.updateUCBEst(armGraph[armPulled][i],reward2)

def armToPullThomp(userArrived,armToPull):
	generated = np.random.beta(userArrived.alpha,userArrived.beta,(N_arms))
	armToPull = argmax(generated)

def armToPullUCB(userArrived,armToPull):
	armToPull = -1
	userArrived.updateUCBTime()
	for i in range(userArrived.num_pulled.size):
		if num_pulled[i] == 0:
			armToPull = i
			break

	if armToPull == -1:
		generated = userArrived.est_mean + userArrived.ucb
		armToPull = argmax(generated)
