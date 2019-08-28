import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

nArms = 2
nTime = 100
nSim = 100

ipolicy = np.zeros((nArms,nArms))
policy = np.zeros((nArms,nArms))
mu = np.zeros((nArms,nArms))
iurn = np.zeros((nArms,1))
urn = np.zeros((nArms,1))
initTotalUrn = 0

iurn[0] = 10	#initialise urn
iurn[1] = 10
initTotalUrn = np.sum(urn)

ipolicy[0,0] = 1	#initialising policy (policy[0,1] = prob of choosing arm 0 given user with preference for arm 1 comes)
ipolicy[0,1] = 0
ipolicy[1,0] = 0.5
ipolicy[1,1] = 0.5

mu[0,0] = 1 # initialise bernoulli reward matrix
mu[0,1] = 0.3
mu[1,0] = 0.4
mu[1,1] = 0.5

arms = range(nArms)

def reset():
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	urn = iurn
	initTotalUrn = np.sum(urn)
	policy = ipolicy
	arms = range(nArms)


def slot(time):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy
	probBall = urn/(initTotalUrn + time)
	armPref = int(np.random.choice(arms, 1, p=probBall[:,0])[0])
	armChosen = int(np.random.choice(arms, 1, p=policy[armPref])[0])
	rewardProb = mu[armPref,armChosen]
	reward = (np.random.binomial(1,rewardProb))
	if reward == 1:
		urn[armChosen] += 1
	else :
		for i in range(nArms):
		 	if i != armChosen:
		 	 	urn[i] += 1/(nArms - 1)
	return [reward,armPref,armChosen,probBall]

def simulate(maxTime,numSim):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy
	avProp = np.zeros((maxTime,nArms))
	for i in range(numSim):
		print(str(i))
		reset()
		prop = np.zeros((maxTime,nArms))
		for j in range(maxTime):
			[d1,d2,d3,d4] = slot(j)
			prop[j] = d4.T

		avProp += prop
	
	avProp = avProp/numSim	
			
	pl.plot(avProp[:,0],'r') 
	pl.plot(avProp[:,1],'g')

	pl.show()


#--------------------------------------------------Main Code Starts----------------------------------------------------------------#


simulate(nTime,nSim)






