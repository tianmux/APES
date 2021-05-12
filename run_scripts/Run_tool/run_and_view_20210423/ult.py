from array import array
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Get the input parameters
def get_input_para(inputfile):
    tempinput = {}
    with open(inputfile) as inputfile:
        for line in inputfile:
            if len(line.split())>1:
                tempinput[line.split()[0]] = line.split()[1:]
        for i in tempinput:
            for j in range(len(tempinput[i])):
                tempinput[i][j] = float(tempinput[i][j])
    return tempinput

# Get the VI data
def get_VI(datafile,n_stride,nTurns,store_step,NpRF,h,nRF,nBeam,omegarf,V0,V0Q,II,IQ):
    test = array('d')
    with open(datafile, mode='rb') as file: # b is important -> binary
        test.fromfile(file,int((n_stride)*nTurns/store_step*NpRF*h[0]))
    time = np.array(test[0::n_stride],dtype='float64')
    Ibi = np.array(test[1::n_stride],dtype='float32')
    #Ibi_1 = np.array(test[2::n_stride])
    Ibi2 = np.array(test[3::n_stride],dtype='float32')

    V = []
    Ig = []
    #U = []
    Vref = []
    Iref = []
    for i in range(nRF):
        V.append(np.array(test[1+2*nBeam+0+i*3::n_stride],dtype='float32'))
        Ig.append(np.array(test[1+2*nBeam+1+i*3::n_stride],dtype='float32'))

        Vref.append(V0[i]*np.sin(omegarf[i]*time)+V0Q[i]*np.cos(omegarf[i]*time))
        Iref.append(II[i]*np.sin(omegarf[i]*time)+IQ[i]*np.cos(omegarf[i]*time))

    return V,Ig,Ibi,Ibi2,Vref,Iref

# Plot VI
def plot_VI(startTurn,startRF,nRF,nRFsamp,NpRF,nBeam,h,V,Vref,Iref,Ig,Ibi,Ibi2,cwd):
    print(nRF)
    rng1 = NpRF*(h[0]*startTurn+startRF)
    rng2 = NpRF*(h[0]*startTurn+startRF+nRFsamp)
    rng3 = 50
    step1 = 1
    step2 = 1

    if nRF>1:
        fig1,axes1 = plt.subplots(4,nRF+1)
        for i in range(nRF):
            axes1[0][i].plot(V[i][rng1:rng2:step1],'r.-',ms=10)
            axes1[0][i].plot((Vref[i])[rng1:rng2:step1],'g.-',ms=10)
            axes1[0][i].axhline(y=0)
            #axes1[0][i].set_xlim([int(NpRF/2)-rng3,int(NpRF/2)+rng3])
            #axes1[0][i].set_ylim([-4e6,4e6])
            axes1[0][0].axhline(y=3.7e6/14*9)
            axes1[0][1].axhline(y=3.7e6/14*5)

            #axes1[0][1].axhline(y=3.7e6/14*6)
            #axes1[0][1].axhline(y=7.02e7)
            axes1[0][i].axvline(x=int(NpRF/2))
            axes1[1][i].axvline(x=int(NpRF/2))
            axes1[1][i].plot(Ibi[rng1:rng2:step1],'r.-',ms=10)
            if nBeam == 2 :
                axes1[1][i].plot(Ibi2[rng1:rng2:step1],'g.-',ms=10)
            #axes1[1][i].set_xlim([int(NpRF/2)-rng3,int(NpRF/2)+rng3])
            #axes1[2][i].set_xlim([int(NpRF/2)-rng3,int(NpRF/2)+rng3])
            axes1[2][i].axhline(y=0)

            axes1[2][i].plot((V[i]-Vref[i])[rng1:rng2:step1],'r.-',ms=10)
            axes1[2][i].axvline(x=int(NpRF/2))
            #axes1[3][i].plot((Ig[i])[rng1:rng2:step1],'r.-',ms=10)
            #axes1[3][i].plot((Iref[i])[rng1:rng2:step1],'g-',ms=10)
            axes1[3][i].plot((Iref[i]-Ig[i])[rng1:rng2:step1],'g-',ms=10)
            axes1[3][i].axvline(x=int(NpRF/2))
            #axes1[0][i].set_ylim([-3e6,3e6])
        Vsum = np.sum(V,0)
        Vref_sum = np.sum(Vref,0)

        axes1[0][nRF].plot(Vsum[rng1:rng2:step1],'r.-',ms=10)
        axes1[0][nRF].plot(Vref_sum[rng1:rng2:step1],'g.-',ms=10)
        #axes1[0][nRF].set_xlim([int(NpRF/2)-rng3,int(NpRF/2)+rng3])
        #axes1[0][nRF].set_ylim([-4e6,4e6])
        axes1[0][nRF].axhline(y=0)
        axes1[0][nRF].axhline(y=3.7e6)
        axes1[1][nRF].axvline(x=int(NpRF/2))
        axes1[0][nRF].axvline(x=int(NpRF/2))
        #axes1[1][nRF].set_xlim([int(NpRF/2)-rng3,int(NpRF/2)+rng3])
        axes1[1][nRF].plot(Ibi[rng1:rng2:step1],'r.-',ms=10)
        if nBeam == 2 :
            axes1[1][nRF].plot(Ibi2[rng1:rng2:step1],'g.-',ms=10)
        axes1[2][nRF].plot((Vsum-Vref_sum)[rng1:rng2:step1],'r.-',ms=10)
        axes1[2][nRF].axvline(x=int(NpRF/2))
        axes1[3][nRF].axvline(x=int(NpRF/2))

    else:
        fig1,axes1 = plt.subplots(4,1)
        for i in range(nRF):
            axes1[0].plot(V[i][rng1:rng2:step1],'rx-',ms=10)
            axes1[0].plot((Vref[i])[rng1:rng2:step1],'g.-',ms=10)
            axes1[0].axhline(y=0)
            
            axes1[0].axvline(x=int(NpRF/2))
            axes1[1].plot(Ibi[rng1:rng2:step1],'r.-',ms=10)
            axes1[1].axvline(x=int(NpRF/2))
            axes1[2].plot((V[i]-Vref[i])[rng1:rng2:step1],'r.-',ms=10)
            axes1[2].axhline(y=0)
            axes1[2].axvline(x = int(NpRF/2))
            axes1[3].plot((Ig[i])[rng1:rng2:step1],'r.-',ms=10)
            axes1[3].plot((Iref[i])[rng1:rng2:step1],'g.-',ms=10)
            axes1[3].axhline(y=0)
            axes1[3].axvline(x=int(NpRF/2))
    fig1.set_figheight(30)
    fig1.set_figwidth(30)
    #print((V[0]-Vref[0])[rng1:rng2:step1][int(NpRF/2)]/1e6)
    fn_VI = os.path.join(cwd,'VI2.pdf')
    plt.savefig(fn_VI,bbox_inches='tight')
    plt.show()