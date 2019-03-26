import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl


N_arms = 3
N_time = 500

alpha = np.zeros(N_arms)
arms = np.zeros(N_arms)
alpha_init = 20
total = alpha_init*N_arms

rewardInfluence = 1
#actualMeans = np.random.uniform(0,1,(N_arms,N_arms))
actualMeans = np.zeros((3,3))
actualMeans[0][0] = 0.7
actualMeans[1][0] = 0.3
actualMeans[2][0] = 0.2

actualMeans[0][1] = 0.1
actualMeans[1][1] = 0.9
actualMeans[2][1] = 0.1

actualMeans[0][2] = 0.3
actualMeans[1][2] = 0.4
actualMeans[2][2] = 0.6

for i in range(N_arms):
	#actualMeans[i][i] = np.random.uniform(0.5,1)
	alpha[i] = alpha_init 
	arms[i] = i

def reset():
	global alpha,total
	global alpha_init
	alpha = np.zeros(N_arms)
	total = alpha_init*N_arms
	alpha_init = 20
	for i in range(N_arms):
		alpha[i] = alpha_init 


def slot(armChosen):
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	prob = np.zeros(N_arms)

	for i in range(N_arms):
		prob[i] = alpha[i]/(total)
	armPref = int(np.random.choice(arms, 1, p=prob)[0])
	if armChosen == armPref:
		reward = rewardInfluence*2*np.random.binomial(1,actualMeans[armChosen][armChosen]) 
		alpha[armPref] += reward
		total += reward	
	else:
		reward = rewardInfluence*2*(np.random.binomial(1,actualMeans[armPref][armChosen]) - 0.5)
		alpha[armChosen] += reward
		alpha[armPref] -= reward
		if alpha[armPref] < 0 :
			diff = 0 - alpha[armPref]
			alpha[armPref] += diff
			alpha[armChosen] -= diff

		if alpha[armChosen] < 0 :
			diff = 0 - alpha[armChosen]
			alpha[armPref] -= diff
			alpha[armChosen] += diff		

	return reward


def sillyPolicy(numSim):
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	alp = np.zeros((N_arms,N_time+1))
	for i in range(N_arms):
		alp[i][0] = numSim/N_arms
	for k in range(numSim):
		reset()
		for i in range(N_time):
			armChosen = 2
			r = slot(armChosen)
			for j in range(N_arms):
					alp[j][i+1] += (alpha[j]/total)
	alp = alp/numSim
	pl.plot(alp[0],'c')
	pl.plot(alp[1],'b')
	pl.plot(alp[2],'g')
	pl.show()


def policy0(numSim):
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	alp = np.zeros((N_arms,N_time+1))
	for i in range(N_arms):
		alp[i][0] = numSim/N_arms
	for k in range(numSim):
		reset()
		avgR = np.ones(N_arms)
		numPulled = np.ones(N_arms)
		p = np.zeros(N_arms)
		
		for i in range(N_time):
			s = 0
			for j in range(N_arms):
				s+=avgR[j]
			p = avgR/s
			armChosen =  int(np.random.choice(arms, 1, p=p)[0])
			r = max(0,slot(armChosen))
			avgR[armChosen] = (avgR[armChosen]*numPulled[armChosen] + r)/(numPulled[armChosen] + 1)
			numPulled[armChosen] += 1
			for j in range(N_arms):
				alp[j][i+1] += (alpha[j]/total)


	alp = alp/numSim
	pl.plot(alp[0],'c')
	pl.plot(alp[1],'b')
	pl.plot(alp[2],'g')
	pl.show()

def policy1():
	global alpha
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	thompA = np.zeros(N_arms)
	prob = np.zeros(N_arms)
	for i in range(N_arms):
		thompA[i] = alpha_init
	tot = alpha_init*N_arms
	for i in range(N_time):
		for i in range(N_arms):
			prob[i] = thompA[i]/tot	
		armChosen =  int(np.random.choice(arms, 1, p=prob)[0])
		r=slot(armChosen)
		thompA[armChosen] += r
		tot += r
		if thompA[armChosen]<0:
			thompA[armChosen] -= r
			tot -= r


def testpolicy():
	
	
#---------------------------------------------------------------------------------------------------------------------


print(actualMeans)
sillyPolicy(100)
policy0(100)

