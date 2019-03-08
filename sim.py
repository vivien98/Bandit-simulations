import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

N_arms = 4
N_users = 5
ucb_alpha = 1
t = 1
actualBernMeans = np.random.uniform(0,1,(N_users,N_arms))

print(actualBernMeans)

class user :
	def __init__(self,nbu,index):
		self.nbu = nbu 					#neighbouring users in graph
		self.index = index						

		self.alpha = np.ones(N_arms)    #beta params
		self.beta = np.ones(N_arms)

		self.est_mean = np.zeros(N_arms)      #ucb algorithm params
		self.ucb = np.zeros(N_arms)
		self.num_pulled = np.zeros(N_arms)
		self.num_chosen = np.zeros(N_arms)

		self.regret = [0]     # regret

	def updateThompDistr(self,arm,bern):
		self.num_pulled[arm] += 1
		self.alpha[arm] = self.alpha[arm] + bern
		self.beta[arm] = self.beta[arm] + 1 - bern				

	def updateUCBEst(self,arm,bern):
		self.num_pulled[arm] += 1
		self.est_mean[arm] = (self.est_mean[arm]*(self.num_pulled[arm] - 1) + bern)/(self.num_pulled[arm])
		self.ucb[arm] = math.sqrt(ucb_alpha*math.log(t) / (2*self.num_pulled[arm]))

	def updateUCBTime(self):
		for i in range(N_arms):
			self.ucb[i] = math.sqrt(ucb_alpha*math.log(t) / (2*self.num_pulled[i]))



def updateThompGraph(users,armGraph,armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived.index][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived.index])-actualBernMeans[userArrived.index][armPulled]
	#print(inst_regret)
	#print(userArrived.index)
	#print(" ")
	temp = userArrived.regret[-1] + inst_regret
	userArrived.regret = np.append(userArrived.regret,temp)

	userArrived.updateThompDistr(armPulled,reward)

	for i in range(userArrived.nbu.size):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateThompDistr(armPulled,reward1)

	for i in range(armGraph[armPulled].size):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived.index][armGraph[armPulled][i]])
		userArrived.updateThompDistr(armGraph[armPulled][i],reward2)

def updateUCBGraph(users,armGraph,armPulled,userArrived):
	reward = np.random.binomial(1,actualBernMeans[userArrived.index][armPulled])
	inst_regret = np.amax(actualBernMeans[userArrived.index])-actualBernMeans[userArrived.index][armPulled]
	temp = userArrived.regret[-1] + inst_regret
	userArrived.regret = np.append(userArrived.regret,temp)

	userArrived.updateUCBEst(armPulled,reward)

	for i in range(userArrived.nbu.size):
		reward1 = np.random.binomial(1,actualBernMeans[userArrived.nbu[i]][armPulled])
		users[userArrived.nbu[i]].updateUCBEst(armPulled,reward1)

	for i in range(armGraph[armPulled].size):
		reward2 = np.random.binomial(1,actualBernMeans[userArrived.index][armGraph[armPulled][i]])
		userArrived.updateUCBEst(armGraph[armPulled][i],reward2)

def armToPullThomp(userArrived,armToPull):
	generated = np.random.beta(userArrived.alpha,userArrived.beta,(N_arms))
	armToPull = np.argmax(generated)
	#print(armToPull)
	return armToPull

def armToPullUCB(userArrived,armToPull):
	armToPull = -1
	userArrived.updateUCBTime()
	for i in range(userArrived.num_pulled.size):
		if userArrived.num_pulled[i] == 0:
			armToPull = i
			break

	if armToPull == -1:
		generated = userArrived.est_mean + userArrived.ucb
		armToPull = np.argmax(generated)

	return armToPull

def simulation(T,users,armGraph,isThompson):
	t = 1
	for i in range(T):
		gen = int(np.random.uniform(0,N_users)) 
		armToPull = -1
		if isThompson == 1:
			armToPull = armToPullThomp(users[gen],armToPull)
			users[gen].num_chosen[armToPull] += 1
			#print(armToPull)
			updateThompGraph(users,armGraph,armToPull,users[gen])
		else:
			armToPull = armToPullUCB(users[gen],armToPull)
			users[gen].num_chosen[armToPull] += 1
			updateUCBGraph(users,armGraph,armToPull,users[gen])

		t += 1

	for i in range(N_users):
		bestarm = np.argmax(users[i].num_chosen)
		print(bestarm)
		pl.plot(users[i].regret)
		pl.show()


#------------------------------------------------------------------------------------------------------------------#


u0 = user(np.array([1,2,4]),0)
u1 = user(np.array([0,1]),1)
u2 = user(np.array([0,1,3]),2)
u3 = user(np.array([2,4]),3)
u4 = user(np.array([0,3]),4)

users = np.array([])

users = np.append(users,[u0,u1,u2,u3,u4])

armGraph = np.array([[1,3],[0,2],[1,3],[0,2]])


simulation(5000,users,armGraph,0)

#----------------------------------------------------------------------------------------------------------#





			