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
	randomArrReci = np.random.rand(nSim,nTime);
	randomArrRewi = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	avgReg = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		regT = np.zeros(nTime)
		prevReg = 0
		p = 0.5
		q = 0.5
		pi = float(B[0,0]+B[0,1]-1 > 0) # p and q were the matrix known
		qi = float(B[1,1]+B[1,0]-1 < 0)
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

				if pi > randomArrReci[i,j]:		# generating arm to be shown for known B case
					armChoseni = 0
				else:
					armChoseni = 1

			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0

				if qi > randomArrReci[i,j]:		# generating arm to be shown for known B case
					armChoseni = 1
				else:
					armChoseni = 0

			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0

			if B[armPref,armChoseni] > randomArrRewi[i,j]: # generating reward
				rewi = 1
			else:
				rewi = 0

			if j<tThresh:
				bEst[armPref,armChosen] += rew
				cnt[armPref,armChosen] += 1
			urni = [0.0,0.0]
			prevZ = urn[0]
			prevZi = urni[0]
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			urni[armChoseni] += rewi
			urni[1-armChoseni] += 1-rewi
			delZ = urn[0] - prevZ
			delZi = urni[0] - prevZi
			propT[j] = prop[0]
			regT[j] = prevReg + delZi - delZ
			prevReg = regT[j]
		avgProp += propT
		avgReg += regT

	avgProp = avgProp/nSim
	avgReg = avgReg/nSim

	return avgProp,avgReg


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
	randomArrReci = np.random.rand(nSim,nTime);
	randomArrRewi = np.random.rand(nSim,nTime);
	avgProp = np.zeros(nTime)
	avgReg = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT = np.zeros(nTime)
		regT = np.zeros(nTime)
		prevReg = 0
		p = 0.5
		q = 0.5
		pi = float(B[0,0]+B[0,1]-1 > 0) # p and q were the matrix known
		qi = float(B[1,1]+B[1,0]-1 < 0)
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

				if pi > randomArrReci[i,j]:		# generating arm to be shown for known B case
					armChoseni = 0
				else:
					armChoseni = 1

			else:
				if q > randomArrRec[i,j]:		# generating arm to be shown 
					armChosen = 1
				else:
					armChosen = 0

				if qi > randomArrReci[i,j]:		# generating arm to be shown for known B case
					armChoseni = 1
				else:
					armChoseni = 0
			if B[armPref,armChosen] > randomArrRew[i,j]: # generating reward
				rew = 1
			else:
				rew = 0

			if B[armPref,armChoseni] > randomArrRewi[i,j]: # generating reward
				rewi = 1
			else:
				rewi = 0

			thompsonFactor = 2
			alpha[armPref,armChosen] += thompsonFactor*rew
			beta[armPref,armChosen] += thompsonFactor*(1 - rew)
			urni = [0.0,0.0]
			prevZ = urn[0]
			prevZi = urni[0]
			urn[armChosen] += rew
			urn[1-armChosen] += 1-rew
			urni[armChoseni] += rewi
			urni[1-armChoseni] += 1-rewi
			delZ = urn[0] - prevZ
			delZi = urni[0] - prevZi
			propT[j] = prop[0]
			regT[j] = prevReg + delZi - delZ
			prevReg = regT[j]

		avgProp += propT
		avgReg += regT


	avgProp = avgProp/nSim
	avgReg = avgReg/nSim

	return avgProp,avgReg


#___________________________________________________________MAIN___________________________________________________________#

b00 = 0.9 # initialise bernoulli reward matrix # 
b01 = 0.7
b10 = 0.7
b11 = 0.9
B = np.matrix([[b00,b01],[b10,b11]])

nSim = 1000
nTime = 1000
initUrn = 20
initProp = 0.5
tThresh = 150

out1 = simulateOPT(nSim,nTime,B,initUrn,initProp)
out2,reg2 = simulateETC(nSim,nTime,B,initUrn,initProp,tThresh)
out3,reg3 = simulateTHO(nSim,nTime,B,initUrn,initProp)


pl.subplot(1,2,1)
pl.plot(out1 , 'r-',label='Optimal policy',linewidth = 2.5)				# POPULATION PROPORTION PLOTTING # NORMAl
pl.plot(out2 ,'b--',label='ETC policy',linewidth = 2.5)
pl.plot(out3 ,'g-.',label='TS policy',linewidth = 2.5)
pl.legend(loc='lower right',frameon=True,prop={"size":20})
pl.xlabel('Time',fontsize=20)
pl.ylabel('Proportion of Type 1 users',fontsize=20)
pl.tick_params(labelsize=20);
pl.subplot(1,2,2)
pl.plot(reg2 ,'b-',label='Regret for ETC policy',linewidth = 2.5)			# REGRET PLOTTING
pl.plot(reg3 ,'g--',label='Regret for TS policy',linewidth = 2.5)
pl.legend(loc='lower right',frameon=True,prop={"size":20})
pl.xlabel('Time',fontsize=20)
pl.ylabel('Cumulative Regret',fontsize=20)
pl.tick_params(labelsize=20);
pl.show()

pl.subplot(1,2,1)
pl.plot(out1 , 'r-',label='Optimal policy',linewidth = 2.5)				# POPULATION PROPORTION PLOTTING #LOG
pl.plot(out2 ,'b--',label='ETC policy',linewidth = 2.5)
pl.plot(out3 ,'g-.',label='TS policy',linewidth = 2.5)
pl.legend(loc='lower right',frameon=True,prop={"size":20})
pl.xlabel('Time',fontsize=20)
pl.ylabel('Proportion of Type 1 users',fontsize=20)
pl.tick_params(labelsize=20);
pl.subplot(1,2,2)
pl.plot(reg2 ,'b-',label='Regret for ETC policy',linewidth = 2.5)			# REGRET PLOTTING LOG
pl.plot(reg3 ,'g--',label='Regret for TS policy',linewidth = 2.5)
pl.legend(loc='upper left',frameon=True,prop={"size":20})
pl.xscale("log")
pl.xlabel('Time',fontsize=20)
pl.ylabel('Cumulative Regret',fontsize=20)
pl.tick_params(labelsize=20);
pl.show()
