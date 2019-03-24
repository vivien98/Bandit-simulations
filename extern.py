import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl


N_arms = 2
N_time = 2000
alpha = np.zeros(N_arms)
arms = np.zeros(N_arms)
alpha_init = 15
empty = np.array([alpha_init])
actualMeans = np.random.uniform(0,1,(N_arms,N_arms))


for i in range(N_arms):
	actualMeans[i][i] = np.random.uniform(0.5,1)
	alpha[i] = alpha_init 
	arms[i] = i

def slot(armChosen):
	prob = np.zeros(N_arms)

	for i in range(N_arms):
		prob[i] = alpha[i]/(alpha_init*N_arms)
	armPref = int(np.random.choice(arms, 1, p=prob)[0])
	if armChosen == armPref:
		reward = 2*np.random.binomial(1,actualMeans[armChosen][armChosen]) 
	else:
		reward = 2*(np.random.binomial(1,actualMeans[armPref][armChosen]) - 0.5)
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

def policy0(numSim):
	for k in range(numSim):
		avgR = np.ones(N_arms)
		numPulled = np.ones(N_arms)
		p = np.zeros(N_arms)
		alp = np.zeros((N_arms,N_time+1))
		for i in range(N_arms):
			alp[i][0] = alpha_init
		for i in range(N_time):
			s = 0
			for j in range(N_arms):
				s+=avgR
			p = avgR/s
			armChosen =  int(np.random.choice(arms, 1, p=p)[0])
			r = slot(armChosen)
			avgR[armChosen] = (avgR[armChosen]*numPulled[armChosen] + r)/(numPulled[armChosen] + 1)
			numPulled[armChosen] += 1
			for j in range(N_arms):
				alp[j][i+1] = alpha[j]
	pl.plot(alp[0],'c')
	pl.plot(alp[1],'b')
	pl.show()


def policy1():
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



#---------------------------------------------------------------------------------------------------------------------


print(actualMeans)
policy0(1)

