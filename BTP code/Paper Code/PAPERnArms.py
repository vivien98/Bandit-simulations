import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

def simulateOPT(nArms,nSim,nTime,B,initUrn):
	avgProp = np.zeros(nTime)
	randomArrRew = np.random.rand(nSim,nTime);
	for i in range(nSim):
		print(i)
		urn = initUrn/nArms * np.ones(nArms)
		propT = np.zeros(nTime)
		Ba = np.ones((nArms,nArms))
		Bb = np.ones((nArms,nArms))
		for j in range(nTime):
			prop = urn/(initUrn+j)
			armPref = np.random.choice(nArms,p=prop)
			Bmat = B
			row1 = 1-Bmat[0]
			row1[0] = Bmat[0,0]
			if armPref == 0:
				armChosen = np.argmax(row1)
			else:
				if 1-Bmat[armPref,0] > (1-Bmat[armPref,armPref])/(nArms-1):
					armChosen = 0
				else:
					armChosen = armPref

			if B[armPref,armChosen]>randomArrRew[i,j] :
				rew = 1
			else:
				rew = 0

			Ba[armPref,armChosen] += rew
			Bb[armPref,armChosen] += 1-rew

			if armChosen == armPref:
				if rew == 1:
					urn[armPref] += 1
				else:
					armInc = armPref
					while armInc == armPref:
						armInc = np.random.choice(nArms)
					urn[armInc] += 1				
			else:
				urn[armChosen] += rew
				urn[armPref] += 1-rew
			propT[j] = prop[0]
		avgProp += propT
	avgProp = avgProp/nSim
	return avgProp

def simulateTHO(nArms,nSim,nTime,B,initUrn):
	avgProp = np.zeros(nTime)
	randomArrRew = np.random.rand(nSim,nTime);
	for i in range(nSim):
		print(i)
		urn = initUrn/nArms * np.ones(nArms)
		propT = np.zeros(nTime)
		Ba = np.ones((nArms,nArms))
		Bb = np.ones((nArms,nArms))
		for j in range(nTime):
			prop = urn/(initUrn+j)
			armPref = np.random.choice(nArms,p=prop)
			Bmat = np.random.beta(Ba,Bb)
			row1 = 1-Bmat[0]
			row1[0] = Bmat[0,0]
			if armPref == 0:
				armChosen = np.argmax(row1)
			else:
				if 1-Bmat[armPref,0] > (1-Bmat[armPref,armPref])/(nArms-1):
					armChosen = 0
				else:
					armChosen = armPref

			if B[armPref,armChosen]>randomArrRew[i,j] :
				rew = 1
			else:
				rew = 0

			Ba[armPref,armChosen] += rew
			Bb[armPref,armChosen] += 1-rew

			if armChosen == armPref:
				if rew == 1:
					urn[armPref] += 1
				else:
					armInc = armPref
					while armInc == armPref:
						armInc = np.random.choice(nArms)
					urn[armInc] += 1				
			else:
				urn[armChosen] += rew
				urn[armPref] += 1-rew
			propT[j] = prop[0]
		avgProp += propT
	avgProp = avgProp/nSim
	return avgProp






#______________________________MAIN___________________________________#


B = np.matrix([[0.9,0.7,0.7,0.7,0.7],[0.7,0.9,0.7,0.7,0.7],[0.7,0.7,0.9,0.7,0.7],[0.7,0.7,0.7,0.9,0.7],[0.7,0.7,0.7,0.7,0.9]])

nArms = 5
nSim = 10
nTime = 1000
initUrn = 50
initProp = 0.5
tThresh = 150

out1 = simulateOPT(nArms,nSim,nTime,B,initUrn)
out2 = simulateTHO(nArms,nSim,nTime,B,initUrn)

pl.plot(out1 , 'r-',label='Optimal policy',linewidth = 2.5)				# POPULATION PROPORTION PLOTTING # NORMAl
pl.plot(out2 , 'b--',label='Thompson sampling',linewidth = 2.5)
pl.legend(loc='lower right',frameon=True,prop={"size":20})
pl.xlabel('Time',fontsize=20)
pl.ylabel('Proportion of Type 1 users',fontsize=20)
pl.tick_params(labelsize=20);
pl.show()