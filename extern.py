import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl


N_arms = 5
N_time = 100000

alpha = np.zeros(N_arms)
arms = np.zeros(N_arms)
alpha_init = 150
initial = alpha_init
total = alpha_init*N_arms
userType = 0

rewardInfluence = 1
#actualMeans = np.random.uniform(0,1,(N_arms,N_arms))
actualMeans = np.zeros((N_arms,N_arms))
actualMeans[0][0] = 0.7
actualMeans[1][0] = 0.1
actualMeans[2][0] = 0.2
actualMeans[3][0] = 0.6
actualMeans[4][0] = 0.2

actualMeans[0][1] = 0.1
actualMeans[1][1] = 0.9
actualMeans[2][1] = 0.05
actualMeans[3][1] = 0.2
actualMeans[4][1] = 0.2

actualMeans[0][2] = 0.3
actualMeans[1][2] = 0.4
actualMeans[2][2] = 0.6
actualMeans[3][2] = 0.2
actualMeans[4][2] = 0.55

actualMeans[0][3] = 0.3
actualMeans[1][3] = 0.4
actualMeans[2][3] = 0.4
actualMeans[3][3] = 0.65
actualMeans[4][3] = 0.2

actualMeans[0][4] = 0.4
actualMeans[1][4] = 0.4
actualMeans[2][4] = 0.5
actualMeans[3][4] = 0.2
actualMeans[4][4] = 0.55

for i in range(N_arms):
	alpha[i] = alpha_init 
	arms[i] = i

def reset():
	global alpha,total
	global alpha_init,initial
	global userType
	alpha = np.zeros(N_arms)
	total = alpha_init*N_arms
	alpha_init = initial
	for i in range(N_arms):
		alpha[i] = alpha_init 


def slot(armChosen):
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	global userType
	prob = np.zeros(N_arms)

	for i in range(N_arms):
		prob[i] = alpha[i]/(total)
	armPref = int(np.random.choice(arms, 1, p=prob)[0])
	if armChosen == armPref:
		reward = rewardInfluence*2*(np.random.binomial(1,actualMeans[armChosen][armChosen]))
		alpha[armPref] += reward
		# for j in range(N_arms):
		# 	alpha[j] -= reward/(N_arms-1)
		# 	if alpha[j] < 0:
		# 		alpha[]
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
	userType = armPref
	return reward


def sillyPolicy(numSim):#pull one arm always
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	alp = np.zeros((N_arms,N_time+1))
	for i in range(N_arms):
		alp[i][0] = numSim/N_arms
	for k in range(numSim):
		reset()
		for i in range(N_time):
			armChosen = 0
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
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def armToPull1(): #when mu is being estimated
	global alpha,total
	global alpha_init
	global N_arms,N_time,rewardInfluence,arms
	arm = int(np.random.choice(arms,1))
	# invertAlpha = np.zeros(N_arms)
	# acc = 0
	# for i in range(N_arms):
	# 	if alpha[i] != 0:
	# 		invertAlpha[i] = 1/alpha[i]
	# 		acc += invertAlpha[i]
	# invertAlpha /= acc
	Alpha = alpha/total
	arm = int(np.random.choice(arms,1,p=Alpha))
		
	return arm

def armToPull2(amu,bestArm): # when mu estimation is over
	if alpha[bestArm]/total > 0.8: #need a way to compute the threshold from the estimated matrix. Try different values of threshold till it fails
		arm = bestArm
	else:
		excludeBestArm = alpha[np.arange(len(alpha))!=bestArm]
		targetArm = np.argmax(excludeBestArm)
		if bestArm <= targetArm:
			targetArm += 1
		excBestArm = amu[targetArm][np.arange(len(amu[targetArm]))!=bestArm]
		arm = np.argmax(excBestArm)
		if bestArm <= arm:
			arm += 1
	return arm

def testpolicy(numSim):			#choose the arm with max alpha as target arm then keep choosing the max mu arm in its row.
	global alpha,total
	global alpha_init,userType
	global N_arms,N_time,rewardInfluence,arms  
	alp = np.zeros((N_arms,N_time+1))
	mu_est = np.zeros(N_arms)
	for i in range(N_arms):
		alp[i][0] = numSim/N_arms
	for k in range(numSim):
		print(k)
		reset()
		mu = np.zeros(N_arms)
		amu = np.zeros((N_arms,N_arms))
		numPulled = np.zeros((N_arms,N_arms)) 
		numMatched = np.zeros(N_arms)
		#pij = np.zeros(N_arms,N_arms)
		for i in range(N_arms):
			amu[i][i] = -10
		for i in range(N_time):
			bestArm = np.argmax(mu)
			#print(100*i/N_time)
			#print(int(bestArm))
			if i <= 550:
				armChosen = int(armToPull1())
			else:
				armChosen = int(armToPull2(amu,bestArm))
							 				   #exploitation even when there is a lower bound above which exploitation is occuring
			reward = slot(armChosen)
			if reward==0 or reward == 2:
				mu[armChosen] = (mu[armChosen]*numMatched[armChosen] + reward)/(numMatched[armChosen] + 1)
				numMatched[armChosen] += 1
			else:
				amu[userType][armChosen] = (amu[userType][armChosen]*numPulled[userType][armChosen] + reward)/(numPulled[userType][armChosen] + 1)
				numPulled[userType][armChosen] += 1

			for j in range(N_arms):
				alp[j][i+1] += (alpha[j]/total)
				#alp[j][i+2] += (alpha[j+1]/total)

	#mu_est = 0.5*mu_est/numSim
	alp = alp/numSim
	#print(mu_est)
	pl.plot(alp[0],'r') #note:colour by row
	pl.plot(alp[1],'g')
	pl.plot(alp[2],'b')
	pl.plot(alp[3],'c')
	pl.plot(alp[4],'m')
	pl.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------


print(actualMeans)
#sillyPolicy(100)
#policy0(100)
testpolicy(1)
