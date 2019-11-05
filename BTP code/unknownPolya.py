import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle

nArms = 2
nTime = 2000
nSim = 10
thresh =500

ipolicy = np.zeros((nArms,nArms))
policy = np.zeros((nArms,nArms))
mu = np.zeros((nArms,nArms))
iurn = np.zeros(nArms)
urn = np.zeros(nArms)
initTotalUrn = 0
showRew = 0

iurn[0] = 10	#initialise urn
iurn[1] = 10
initTotalUrn = 20

mu[0,0] = 0.8 # initialise bernoulli reward matrix
mu[0,1] = 0.1
mu[1,0] = 0.1
mu[1,1] = 0.8

state = 0

# mu[0,0] = 0.8   #special case when 1010 policy gives more reward than greedy
# mu[0,1] = 0.05
# mu[1,0] = 0.7
# mu[1,1] = 0.7

arms = range(nArms)

def reset():
	global urn,policy,mu,muEst,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muTot
	urn = iurn
	
	muTot = np.zeros((nArms,nArms))
	initTotalUrn = np.sum(urn)
	policy = ipolicy
	arms = range(nArms)
	state = 0



def slot(time,policyID):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muEst,muTot
	probBall = urn/(initTotalUrn + time)
	armPref = int(np.random.choice(arms, 1, p=probBall[:,0])[0])
	if policyID == 0:
		armChosen = int(np.random.choice(arms, 1, p=policy[armPref])[0])
	if policyID == 1:
		armChosen = int(prePolicy1(armPref,probBall,time))
	
	rewardProb = mu[armPref,armChosen]
	reward = (np.random.binomial(1,rewardProb))
	if reward == 1:
		urn[armChosen] += 1
	else :
		for i in range(nArms):
		 	if i != armChosen:
		 	 	urn[i] += 1/(nArms - 1)
	if policyID == 1:
		postPolicy1(armPref,armChosen,reward,probBall,time)
	return [reward,armPref,armChosen,probBall]

def slot1(time,armPref,armChosen):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muEst,muTot
	rewardProb = mu[armPref,armChosen]
	reward = (np.random.binomial(1,rewardProb))
	if reward == 1:
		urn[armChosen] += 1
	else :
		for i in range(nArms):
		 	if i != armChosen:
		 	 	urn[i] += 1/(nArms - 1)
	return reward


def simulate(maxTime,numSim,policyID):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muEst,muTot
	avProp = np.zeros((maxTime,nArms))
	avRew = np.zeros(maxTime)
	cumRew = 0
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
		if showRew == 1:
			avRew += rew
	
	avProp = avProp/numSim	
	if showRew == 1:
		avRew = avRew/numSim
		cumRew = np.cumsum(avRew)
		

	return avProp,avRew,cumRew

def greedyPolicy(armPref):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muEst,muTot
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
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,muEst,muTot
	p = int(mu[0,0] > 1 - mu[0,1])
	q = int(mu[1,1] < 1 - mu[1,0])
	ipolicy[0,0] = p
	ipolicy[1,1] = q
	ipolicy[1,0] = 1-q
	ipolicy[0,1] = 1-p
	if armPref == 0:
		armChosen = 1-p
	else:
		armChosen = q
	return armChosen
#______________________________________________________Policy 1______________________________________________________________________________


def policy1(nSim,thresh,maxTime):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,state
	avProp = np.zeros((nArms,maxTime))
	avRew = np.zeros(maxTime)
	avLast = 0
	cumRew = 0
	
	for i in range(nSim):
		print(str(i))
		reset()
		urn[0] = 10	#initialise urn
		urn[1] = 10
		initTotalUrn = 20
		prop = np.zeros((nArms,maxTime))
		rew = np.zeros(maxTime)
		muEst = np.zeros((nArms,nArms))
		muTot = np.zeros((nArms,nArms))
		p = 1
		q = 1
		last = 0
		for j in range(maxTime):
			probBall = urn/(initTotalUrn + j)
			armPref = int(np.random.choice(arms, 1, p=probBall)[0])
			if j < thresh:
				armChosen = int(np.random.choice(arms, 1, p=[0.5,0.5])[0])
			else:
				if j == thresh:
					muEst = np.divide(muEst,muTot)
					#print(muEst)
					p = int(muEst[0,0] > 1 - muEst[0,1])
					q = int(muEst[1,1] < 1 - muEst[1,0])
					if j == 0:
						p = 1
						q = 1
					#print("p = " + str(p) + ", q = " + str(q))
				if armPref == 0:
					armChosen = 1-p
				else:
					armChosen = q
			rew[j] = slot1(j,armPref,armChosen)
			prop[:,j] = probBall
			if j == maxTime-1:
				last = probBall[0]
			if j < thresh:
				muEst[armPref,armChosen] += rew[j]
				muTot[armPref,armChosen] += 1
		avLast += last
		avProp += prop
		avRew += rew
	avLast /= nSim
	avProp /= nSim
	avRew /= nSim
	pl.plot(avProp[0,:],'b')
	return avLast

def policy1Opt(nSim,thresh,maxTime):
	global urn,policy,mu,nArms,nTime,initTotalUrn,arms,ipolicy,iurn,state
	avLast = 0
	for i in range(nSim):
		reset()
		urn[0] = 10	#initialise urn
		urn[1] = 10
		initTotalUrn = 20
		muEst = np.zeros((nArms,nArms))
		muTot = np.zeros((nArms,nArms))
		p = 1
		q = 1
		last = 0
		for j in range(maxTime):
			probBall = urn/(initTotalUrn + j)
			armPref = int(np.random.choice(arms, 1, p=probBall)[0])
			if j < thresh:
				armChosen = int(np.random.choice(arms, 1, p=[0.5,0.5])[0])
			else:
				if j == thresh:
					muEst = np.divide(muEst,muTot)
					p = int(muEst[0,0] > 1 - muEst[0,1])
					q = int(muEst[1,1] < 1 - muEst[1,0])
					if j == 0:
						p = 1
						q = 1
				if armPref == 0:
					armChosen = 1-p
				else:
					armChosen = q
			rew = slot1(j,armPref,armChosen)
			if j == maxTime-1:
				last = probBall[0]
			if j < thresh:
				muEst[armPref,armChosen] += rew
				muTot[armPref,armChosen] += 1
		avLast += last
	avLast /= nSim
	#print("Prop at end is " + str(avLast))
	return avLast
#____________________________________optimization graph________		
def optimizePolicy(nSim,jump,upper,deadline):
	numItr = int(upper/jump)
	xChart = np.zeros(numItr-1)
	yChart = np.zeros(numItr-1)
	print(numItr)
	for i in range(numItr):
		if i != 0:
			dpt = policy1Opt(nSim,i*jump,deadline)
			xChart[i-1] = i*jump
			yChart[i-1] = dpt
			print(str(i) + " : " + str(xChart[i-1]) + " : " + str(dpt))
	pl.plot(xChart,yChart)


#--------------------------------------------------Main Code Starts----------------------------------------------------------------#


#______Optimize Code____________#
jump = 10
upper = 300
nTime = 500
nSim = 100
optimizePolicy(nSim,jump,upper,nTime)
pl.show()


#______Graph code_________________#
# nSim = 100
# thresh = 60
# nTime = 500
# print(str(thresh))
# policy1(nSim,thresh,nTime)
# pl.show()



































































#________________________________________________________________old code _______________________________________________________#
# greedyPolicy(0)
# [greedyProp,greedyRew,greedyCumRew] = simulate(nTime,nSim,0)



# optimalPolicy(0)
# [optProp,optRew,optCumRew] = simulate(nTime,nSim,0)

# # optimalPolicy(0)

# [longProp,longRew,longCumRew] = simulate(nTime,nSim,1)

# pl.plot(greedyProp[:,0],'r') 
# pl.plot(optProp[:,0],'g') 
# pl.plot(longProp[:,0],'b')

#policy1(nSim,thresh,nTime)

# if showRew == 1 :
# 	pl.plot(greedyRew,'r')
# 	pl.plot(optRew,'b') 
# 	# pl.plot(longRew,'g')
# 	pl.show()
# 	pl.plot(greedyCumRew,'r') 
# 	pl.plot(optCumRew,'b') 
# 	# pl.plot(longCumRew,'g')
# 	pl.show()


