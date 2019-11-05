# BTP code for TWO ARMS ONLY
import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

def simulateOPT(nSim,nTime,B,initUrn,initProp):
	randomArrPref = np.random.rand(nSim,nTime);
	randomArrRec = np.random.rand(nSim,nTime);
	randomArrRew = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		p = float(B[0,0]+B[0,1]-1 > 0)
		q = float(B[1,1]+B[1,0]-1 < 0)
		for j in range(nTime):
			prop = urn/(initUrn+j)
			if prop[0] > randomArrPref[i,j]:   # generating user with some preference
				armPref = 0
			else:
				armPref = 1
			if armPref==0:
				if p > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 0
				else:
					armChosen = 1
			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0
			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			propT[j] = prop[0]

		avgProp += propT

	avgProp = avgProp/nSim

	return avgProp



def simulateETC(nSim,nTime,B,initUrn,initProp,tThresh):
	randomArrPref = np.random.rand(nSim,nTime);
	randomArrRec = np.random.rand(nSim,nTime);
	randomArrRew = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		p = 0.5
		q = 0.5
		bEst = np.zeros((2,2),dtype="float")
		cnt = np.zeros((2,2),dtype="float")
		for j in range(nTime):
			prop = urn/(initUrn+j)
			if prop[0] > randomArrPref[i,j]:   # generating user with some preference
				armPref = 0
			else:
				armPref = 1

			if j < tThresh:						# explore part
				p = 0.5
				q = 0.5
			else:
				if j==tThresh:
					bEst = np.divide(bEst,cnt)
					p = float(bEst[0,0]+bEst[0,1]-1 > 0)
					q = float(bEst[1,1]+bEst[1,0]-1 < 0)

			if armPref==0:
				if p > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 0
				else:
					armChosen = 1
			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0

			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0

			if j<tThresh:
				bEst[armPref,armChosen] += rew
				cnt[armPref,armChosen] += 1
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			propT[j] = prop[0]

		avgProp += propT

	avgProp = avgProp/nSim

	return avgProp


def simulateUCB(nSim,nTime,B,initUrn,initProp,polyAnn):
	randomArrPref = np.random.rand(nSim,nTime);
	randomArrRec = np.random.rand(nSim,nTime);
	randomArrRew = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		bEst = np.zeros((2,2),dtype="float")
		bSum = np.zeros((2,2),dtype="float")
		bUCB = np.zeros((2,2),dtype="float")
		cnt = np.zeros((2,2),dtype="float") + 0.01
		for j in range(nTime):
			prop = urn/(initUrn+j)
			p = 0.5
			q = 0.5
			if polyAnn == 0:
				bUCB = np.sqrt(np.divide(np.log(j+2),32*cnt))
			bEst = np.divide(bSum,cnt)
			if prop[0] > randomArrPref[i,j]:   # generating user with some preference
				armPref = 0
			else:
				armPref = 1

			if bEst[0,0]+bUCB[0,0]+bEst[0,1]+bUCB[0,1] < 1:  # UCB rules to choose arm recommended
				p = 0
			if bEst[0,0]-bUCB[0,0]+bEst[0,1]-bUCB[0,1] > 1:
				p = 1
			if bEst[1,1]-bUCB[1,1]+bEst[1,0]-bUCB[1,0] > 1:
				q = 0
			if bEst[1,1]+bUCB[1,1]+bEst[1,0]+bUCB[1,0] < 1:
				q = 1

			if armPref==0:
				if p > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 0
				else:
					armChosen = 1
			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0
			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0
			bSum[armPref,armChosen] += rew
			cnt[armPref,armChosen] += 1
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			propT[j] = prop[0]

		avgProp += propT

	avgProp = avgProp/nSim

	return avgProp

def simulateTHO(nSim,nTime,B,initUrn,initProp):
	randomArrPref = np.random.rand(nSim,nTime);
	randomArrRec = np.random.rand(nSim,nTime);
	randomArrRew = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		p = 0.5
		q = 0.5
		alpha = np.ones((2,2))
		beta = np.ones((2,2))
		for j in range(nTime):
			prop = urn/(initUrn+j)
			if prop[0] > randomArrPref[i,j]:   # generating user with some preference
				armPref = 0
			else:
				armPref = 1

			sampleMat = np.random.beta(alpha,beta)  # sampling a matrix
			if(sampleMat[0,0] + sampleMat[0,1] - 1 > 0):
				p = 1
			else:
				p = 0
			if(sampleMat[1,1] + sampleMat[1,0] - 1 < 0):
				q = 1
			else:
				q = 0

			if armPref==0:
				if p > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 0
				else:
					armChosen = 1
			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0
			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0

			thompsonFactor = 2
			alpha[armPref,armChosen] += thompsonFactor*rew
			beta[armPref,armChosen] += thompsonFactor*(1 - rew)
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			propT[j] = prop[0]

		avgProp += propT

	avgProp = avgProp/nSim

	return avgProp


#___________________________________________________________MAIN___________________________________________________________#

b00 = 0.9 # initialise bernoulli reward matrix
b01 = 0.3
b10 = 0.1
b11 = 0.6
B = np.matrix([[b00,b01],[b10,b11]])

nSim = 100
nTime = 1000
initUrn = 20
initProp = 0.5
tThresh = 50


out1 = simulateETC(nSim,nTime,B,initUrn,initProp,tThresh)
out2 = simulateOPT(nSim,nTime,B,initUrn,initProp)
out3 = simulateUCB(nSim,nTime,B,initUrn,initProp,0)
out4 = simulateTHO(nSim,nTime,B,initUrn,initProp)

pl.plot(out1 , 'r')
pl.plot(out2 ,'b')
pl.plot(out3 ,'g')
pl.plot(out4 ,'c')
pl.show()	
