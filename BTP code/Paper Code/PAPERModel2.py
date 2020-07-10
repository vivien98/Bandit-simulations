import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

def simulateOPT(nSim,nTime,B,initUrn,initProp):
	randomArrPref1 = np.random.rand(nSim,nTime);
	randomArrRec1 = np.random.rand(nSim,nTime);
	randomArrRew1 = np.random.rand(nSim,nTime);
	randomArrPref2 = np.random.rand(nSim,nTime);
	randomArrRec2 = np.random.rand(nSim,nTime);
	randomArrRew2 = np.random.rand(nSim,nTime);
	avgProp1 = np.zeros(nTime)
	avgProp2 = np.zeros(nTime)
	for i in range(nSim):
		print(str(i))
		urn1 = np.array([initUrn*initProp,initUrn*(1-initProp)])
		urn2 = np.array([initUrn*initProp,initUrn*(1-initProp)])
		propT1 = np.zeros(nTime)
		propT2 = np.zeros(nTime)
		p = float(B[0,0]+B[0,1]-1 > 0)
		q = float(B[1,1]+B[1,0]-1 < 0)
		for j in range(nTime):
			prop1 = urn1/(initUrn+j)
			prop2 = urn2/(initUrn)
			if prop1[0] > randomArrPref1[i,j]:   # generating user with some preference
				armPref1 = 0
			else:
				armPref1 = 1

			if prop2[0] > randomArrPref2[i,j]:   # generating user with some preference
				armPref2 = 0
			else:
				armPref2 = 1

			if armPref1==0:
				if p > randomArrRec1[i,j]:		# generating arm to be shown 
					armChosen1 = 0
				else:
					armChosen1 = 1
			else:
				if q > randomArrRec1[i,j]:		# generating arm to be shown 
					armChosen1 = 1
				else:
					armChosen1 = 0

			if armPref2==0:
				if p > randomArrRec2[i,j]:		# generating arm to be shown 
					armChosen2 = 0
				else:
					armChosen2 = 1
			else:
				if q > randomArrRec2[i,j]:		# generating arm to be shown 
					armChosen2 = 1
				else:
					armChosen2 = 0

			if B[armPref1,armChosen1] > randomArrRew1[i,j]: # generating reward
				rew1 = 1
			else:
				rew1 = 0
			if B[armPref2,armChosen2] > randomArrRew2[i,j]: # generating reward
				rew2 = 1
			else:
				rew2 = 0

			urn1[armChosen1] += rew1
			urn1[1-armChosen1] += 1-rew1
			if armChosen2 == armPref2:
				if rew2 == 0:
					urn2[armChosen2] -= 1
					urn2[1-armChosen2] += 1
			else:
				if rew2 == 1:
					urn2[armChosen2] += 1
					urn2[armPref2] -= 1

			if urn2[0] == -1:
				urn2[0] += 1
				urn2[1] -= 1
			if urn2[1] == -1:
				urn2[1] += 1
				urn2[0] -= 1


			propT1[j] = prop1[0]
			propT2[j] = prop2[0]

		avgProp1 += propT1
		avgProp2 += propT2

	avgProp1 = avgProp1/nSim
	avgProp2 = avgProp2/nSim

	return avgProp1,avgProp2





#_________________MAIN____________________________________________________________________________________________________________________________________





b00 = 0.9 # initialise bernoulli reward matrix # 
b01 = 0.7
b10 = 0.7
b11 = 0.9
B = np.matrix([[b00,b01],[b10,b11]])

nSim = 1000
nTime = 1000
initUrn = 50
initProp = 0.5
tThresh = 150

out1,out2 = simulateOPT(nSim,nTime,B,initUrn,initProp)

pl.plot(out1 , 'r-',label='Model 1',linewidth = 2.5)				# POPULATION PROPORTION PLOTTING # NORMAl
pl.plot(out2 ,'b--',label='Model 2',linewidth = 2.5)
pl.legend(loc='lower right',frameon=True,prop={"size":20})
pl.xlabel('Time',fontsize=20)
pl.ylabel('Proportion of Type 1 users',fontsize=20)
pl.tick_params(labelsize=20);
pl.show()
