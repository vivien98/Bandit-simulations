import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle

nArms = 2
nTime = 10000
nSim = 10

ipolicy = np.zeros((nArms,nArms))
policy = np.zeros((nArms,nArms))
mu = np.zeros((nArms,nArms))
iurn = np.zeros((nArms,1))
urn = np.zeros((nArms,1))
initTotalUrn = 0

iurn[0] = 10	#initialise urn
iurn[1] = 10
initTotalUrn = 20

mu[0,0] = 0.8 # initialise bernoulli reward matrix
mu[0,1] = 0.1
mu[1,0] = 0.5
mu[1,1] = 0.7

state = 0

# mu[0,0] = 0.8   #special case when 1010 policy gives more reward than greedy
# mu[0,1] = 0.05
# mu[1,0] = 0.7
# mu[1,1] = 0.7

arms = range(nArms)

def reset():
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	urn = iurn
	initTotalUrn = np.sum(urn)
	policy = ipolicy
	arms = range(nArms)
	state = 0


def slot(time,policyID):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	probBall = urn/(initTotalUrn + time)
	armPref = int(np.random.choice(arms, 1, p=probBall[:,0])[0])
	if policyID == 0:
		armChosen = int(np.random.choice(arms, 1, p=policy[armPref])[0])
	if policyID == 1:
		armChosen = int(policy1(armPref,probBall))
	
	rewardProb = mu[armPref,armChosen]
	reward = (np.random.binomial(1,rewardProb))
	if reward == 1:
		urn[armChosen] += 1
	else :
		for i in range(nArms):
		 	if i != armChosen:
		 	 	urn[i] += 1/(nArms - 1)
	return [reward,armPref,armChosen,probBall]


def simulate(maxTime,numSim,policyID):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	avProp = np.zeros((maxTime,nArms))
	avRew = np.zeros(maxTime)
	for i in range(numSim):
		print(str(i))
		reset()
		urn[0] = 10	#initialise urn
		urn[1] = 10
		initTotalUrn = 20
		prop = np.zeros((maxTime,nArms))
		rew = np.zeros(maxTime)
		for j in range(maxTime):
			[d1,d2,d3,d4] = slot(j,policyID)
			prop[j] = d4.T
			rew[j] = d1

		avProp += prop
		avRew += rew
	
	avProp = avProp/numSim	
	avRew = avRew/numSim
	cumRew = np.cumsum(avRew)	

	return avProp,avRew,cumRew

def greedyPolicy(armPref):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	p = 1
	q = 1
	ipolicy[0,0] = p
	ipolicy[1,1] = q
	ipolicy[1,0] = 1-q
	ipolicy[0,1] = 1-p
	if armPref == 0:
		armChosen = 1-p
	else:
		armChosen = q
	return armChosen

def optimalPolicy(armPref):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn
	p = int(mu[0,0] > 1 - mu[0,1])
	q = int(mu[1,1] > 1 - mu[1,0])
	ipolicy[0,0] = p
	ipolicy[1,1] = q
	ipolicy[1,0] = 1-q
	ipolicy[0,1] = 1-p
	if armPref == 0:
		armChosen = 1-p
	else:
		armChosen = q
	return armChosen

def policy1(armPref,probBall):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,state
	limit = (ipolicy[1,0]*mu[1,0] + ipolicy[1,1]*(1-mu[1,1]))/(1 + ipolicy[1,0]*mu[1,0] + ipolicy[1,1]*(1-mu[1,1]) -(ipolicy[0,0]*mu[0,0] + ipolicy[0,1]*(1-mu[0,1]))) - 0.04
	# if probBall[0] > limit :
	# 	state += 1
	# 	if state > 250:
	# 		if armPref == 0:
	# 			armChosen = 0
	# 		else:
	# 			armChosen = 1
	# 	else:
	# 		armChosen = optimalPolicy(armPref)
	# else:
	# 	state = 0
	# 	armChosen = optimalPolicy(armPref)
	if probBall[0] > limit or np.sum(urn) > 1500:
		if armPref == 0:
			armChosen = 0
		else:
			armChosen = 1
	else:
		if armPref == 0:
			armChosen = 1
		else:
			armChosen = 0
		

	return armChosen

#--------------------------------------------------Main Code Starts----------------------------------------------------------------#
greedyPolicy(0)
[greedyProp,greedyRew,greedyCumRew] = simulate(nTime,nSim,0)



# optimalPolicy()
# [optProp,optRew,optCumRew] = simulate(nTime,nSim,0)

optimalPolicy(0)

[longProp,longRew,longCumRew] = simulate(nTime,nSim,1)

pl.plot(greedyProp[:,0],'r') 
#pl.plot(optProp[:,0],'b') 
pl.plot(longProp[:,0],'g')
pl.show()
pl.plot(greedyRew,'r')
#pl.plot(optRew,'b') 
pl.plot(longRew,'g')
pl.show()
pl.plot(greedyCumRew,'r') 
#pl.plot(optCumRew,'b') 
pl.plot(longCumRew,'g')
pl.show()


