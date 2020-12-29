import numpy as np
import struct
import sys
from array import array
import itertools
import os
import sys
from scipy import signal
from scipy.optimize import curve_fit
import subprocess
import shutil

pi = np.pi
clight = 299792458
E0Au = 196.9665687*931.5e6
E0Elec = 0.51099895000e6

working_folder = 'bench_marking/1RF_thetaL'
home = os.getcwd()
cwd = os.path.join(home,working_folder)
try:
    os.mkdir(cwd)
except OSError:
    print ("Creation of the directory %s failed" % cwd)
else:
    print ("Successfully created the directory %s" % cwd)
tempinput = {}

        
# Default input
type_of_particle =  1.0 
csv_out = 0
dynamicOn = 0
n_per_show = 10
turn_start = 0

mainRF = 0
main_detune = 0
detune_slow_factor = 1.0 
R = 610.1754 
GMTSQ = 961.0 
Gamma = 19600.0 

nBeam = 1
beam_shift = 94

#----------------------------#
#important inputs
n_turns = 10000 
n_dynamicOn = 3000
n_bunches = 630 
n_fill = 1000 
n_q_ramp = 2000 
n_detune_start = 1000 
n_detune_ramp = 3000 
n_detune_ramp_tot = 3000 
n_I_ramp_start = 1000 
n_I_ramp_end = 3000 
step_store = 1000 
Prad = 9994571.44212 
t_rad_long=  0.03555e10 
Ek_damp = 500000000.0 
Npar = 1 
NperBunch = 3.18e11
N_bins = 33 
fill_step = 12.0 
siglong = 11.368 
A = 1.0 

nRF = 1 
nRF1 = 1
nRFc = 0
nRF2 = 0
nHOM = 0
nCav = np.array([14])
h = np.array([7560])
RoQ = np.array([73])*nCav
delay = np.array([0])
n_fb_on = np.array([50.0])
gII = np.array([0.0])
gQQ = np.array([0.0])
gIQ = np.array([0.0])
gQI = np.array([0.0])
gIIi = np.array([0.0])
gQQi = np.array([0.0])
gIQi = np.array([0.0])
gQIi = np.array([0.0])
PA_cap = np.array([1.0])

#----------------------------#
# the following parameters need to be derived from the input parameters
# here just show some place holder.
QL = np.array([329321.8554911])
Vref_I = np.array([52675398.062246])
Vref_Q = np.array([-2370878.834453])
Iref_I = np.array([0.4869135928499])
Iref_Q = np.array([-0.02191560337399])
I_I_ref_ini = np.array([0.243456796425])
I_I_ref_final = np.array([0.486913592849])
I_Q_ref_ini = np.array([-0.0109578016869])
I_Q_ref_final = np.array([-0.02191560337399])
detune = np.array([0.0])
detune_ini = np.array([0.0])
detune_mid = np.array([-9970.53659004])
detune_final = np.array([-19941.0731801])



beta = np.sqrt(1-1/Gamma**2)
T0 = 2*np.pi*R/(clight*beta)
f0 = 1/T0
if int(type_of_particle==2):
    atomicZ = 79
    Ek = Gamma*E0Au
else:
    atomicZ =1
if int(type_of_particle==1):  
    Ek = Gamma*E0Elec

eta = 1/GMTSQ-1/Gamma**2
if nRF == 1:
    Qs = np.sqrt(h[int(mainRF)]*atomicZ*np.abs(Vref_I[int(mainRF)])*eta/(2*np.pi*Ek))
elif nRF != 1 :
    Qs = np.sqrt(h[int(mainRF)]*atomicZ*np.abs(Vref_I[int(mainRF)]+Vref_I[1])*eta/(2*np.pi*Ek))

omegarf = 2*np.pi*(np.array(h)*f0)
omegac = 2*np.pi*(np.array(h)*f0+detune_final)
Trf = 2*np.pi/omegarf
Rsh = [RoQ[i]*QL[i] for i in range(int(nRF))]

Th = 2*np.pi/omegarf[0]
dthat =Th/N_bins
bucket_height = 2*Qs/(h[mainRF]*eta)*Gamma

print(bucket_height)
print(Ek)
print(Qs)

# setup the current samples

N_samples = 1 
iMin = 2.5
iMax = 2.5
nParMin = iMin/f0/n_bunches/1.6e-19
nParMax = iMax/f0/n_bunches/1.6e-19

# setup the loading angle samples, for the focusing cavity only.
N_thetaL = 11
ThetaL_min = np.zeros(nRF)#17.5
ThetaL_max = np.zeros(nRF)#17.5

ThetaL_min[0] = -5 
ThetaL_max[0] = 6 

dnPar = (nParMax-nParMin)/N_samples
dThetaL = (ThetaL_max-ThetaL_min)/N_thetaL

print("Generating input parameters...")
for charge_factor in range(N_samples):
    for thetaL_factor in range(N_thetaL):
        for i in range(nRF):
            gII[i] = 0
            gQQ[i] = 0
        nPar0 = NperBunch
        Prad0 = 9e6
        nPar = nParMin+dnPar*charge_factor #nPar0/N_samples*(charge_factor+1)
        Prad = Prad0*nPar/nPar0
        if nRF == 1:
            NC = nCav[0] #+nCav[1]
            NF = nCav[0]
            ND = 0
        elif nRF ==2 :
            NC = nCav[0]+nCav[1]
            NF = nCav[0]
            ND = nCav[1]        
        thetaL = np.zeros(nRF)
        Vs = np.zeros(nRF)
        Vq = np.zeros(nRF)
        Phis = np.zeros(nRF)
        PhisPhasor = np.zeros(nRF)
        PhiIQ = np.zeros(nRF)
        PhiIQIg = np.zeros(nRF)
        VrefI = np.zeros(nRF)
        VrefQ = np.zeros(nRF)
        Vreftot = np.zeros(nRF)
        IrefI = np.zeros(nRF)
        IrefQ = np.zeros(nRF)
        QL = np.zeros(nRF)
        Rsh = np.zeros(nRF)
        
        # Need to calculate the required voltage and phase
        # then calculate the inputs for the code, namely VrefI, VrefQ, IrefI, IrefQ.
        
        # for fundamental, 
        Vtot = 23.7e6 # total voltage, just for calcualating the required voltages.
        Urad0 = Prad/(n_bunches*nPar*1.6e-19*f0) # radiation caused Voltage total, depends on beam kinetic energy
        U_loss = Urad0/NC 
        print("Urad0 : ",Urad0)
        V0 = Vtot/NC # old voltage per cavity.
        
        # this is the required real voltage on beam, to compensate radiation loss
        Vsynch_need = U_loss
        # this is the required imaginary voltage, to provide bucket. 
        Vquard_need = V0*np.sin(np.arccos(U_loss/V0))*7560/h[0] # calculating it from known parameters
        
        
        # new cavity voltage per cavity, 
        # once the total voltage and phasor phase of the focusing cavity is chosen, 
        # the result should be decided.
        if nRF ==1 :
            Vnew = np.sqrt(Vsynch_need**2+(NC/(NF-ND)*Vquard_need)**2) 
            Vs[0] = Vsynch_need
            Vq[0] = NC/(NF-ND)*Vquard_need
            #PhisPhasor = np.arctan(Vq[0]/Vs[0])
        if nRF == 2 :
            Vnew = 5861889.40948 # per cavity
            PhisPhasor = 87.4858055644/180*pi
            Vs[0] = Vnew*np.cos(PhisPhasor)
            Vq[0] = Vnew*np.sin(PhisPhasor)
            Vs[1] = (Urad0-Vs[0]*NF)/ND
            Vq[1] = (Vquard_need*(NF+ND)-Vq[0]*NF)/ND
        if nRF == 1:
            Qs = np.sqrt(h[int(mainRF)]*atomicZ*np.abs(Vq[0]*NF)*eta/(2*np.pi*Ek))
        elif nRF != 1 :
            Qs = np.sqrt(h[int(mainRF)]*atomicZ*np.abs(Vq[0]*NF-Vq[1]*ND)*eta/(2*np.pi*Ek))

        print("Vs,Vq: ",Vs,Vq)
        print("Qs = ",Qs)
        PhisPhasor = np.arctan(Vq/Vs)
        Pbeam0 = Prad0/NC # beam power per fundamental cavity, 
        Pbeam = Prad0*nPar/nPar0/NC # beam power per fundamental cavity
        
        IbDC = nPar*f0*1.6e-19*n_bunches
        Pbeam = IbDC*Vs
        print("Beam power per cavity: ",Pbeam)
        
        IbDC = n_bunches*nPar*1.6e-19*f0
        f = h*f0
        # convert RoQ from total to per cavity
        RoQ = RoQ/nCav
        RoQacc = RoQ*2
        print("RoQ per cavity: ", RoQ)
        print("Number of cavity : ",nCav)
        
        Vreftot = np.sqrt(Vs**2+Vq**2) 
        Qbeam0 = Vreftot**2/(RoQacc*Pbeam)
        Qbeam = Qbeam0
        QL = Qbeam
        Rsh = RoQ*QL
        
    # Now calculate the inputs
        thetaL = (ThetaL_min+dThetaL*thetaL_factor)/180.0*pi  # angle between Ig and Vc
        Vbr = 2*IbDC*Rsh
        Vgr = Vreftot/np.cos(thetaL)*(1+Vbr/Vreftot*np.cos(PhisPhasor))
        
        tgPhi = -(Vbr*np.sin(PhisPhasor)/Vreftot+(1+Vbr*np.cos(PhisPhasor)/Vreftot)*np.tan(thetaL))
        tgPhi_ini = -np.tan(thetaL)
        delta_f_ini = f*(tgPhi_ini/2/QL+np.sqrt((tgPhi_ini/2/QL)**2+1))-f
        delta_f = f*(tgPhi/2/QL+np.sqrt((tgPhi/2/QL)**2+1))-f
        
        VrefI = Vreftot*np.sin(PhisPhasor)
        VrefQ = -Vreftot*np.cos(PhisPhasor)

        I_I = Vgr/Rsh*np.sin(PhisPhasor+thetaL) 
        I_Q = -Vgr/Rsh*np.cos(PhisPhasor+thetaL)
        
        I_I_ini = Vreftot/(Rsh)/np.cos(thetaL)*np.sin(PhisPhasor+thetaL)
        I_Q_ini = -Vreftot/(Rsh)/np.cos(thetaL)*np.cos(PhisPhasor+thetaL)
        
        print("Vnew : ",Vreftot)
        print("QL : ",QL)
        print("ThetaL : ",thetaL/pi*180, " [degree]")
        
        print("Tan(PhisPhasor) : ",np.tan(PhisPhasor))
        print("PhisPhasor : ",PhisPhasor/pi*180)

        print("detune tan : ", tgPhi)
        print("detune angle : ", np.arctan(tgPhi)/pi*180, " [degree]")
        print("delta_f : ",delta_f, " [Hz]")
        print("VrefI : ",VrefI)
        print("VrefQ : ",VrefQ)
        
        print("II : ",I_I)
        print("IQ : ",I_Q)
        print("II_ini : ",I_I_ini)
        print("IQ_ini : ",I_Q_ini)

        print("VrefTot : ",Vreftot)
        print("IbDC : ", IbDC)
        
        # now write to the input file
        tempinput['type'] = np.array([type_of_particle])
        tempinput['csv_out'] = np.array([csv_out])
        tempinput['dynamicOn'] = np.array([dynamicOn])
        tempinput['n_per_show'] = np.array([n_per_show])
        tempinput['turn_start'] = np.array([turn_start])
        tempinput['mainRF'] = np.array([mainRF])
        tempinput['main_detune'] = np.array([main_detune])
        tempinput['detune_slow_factor'] = np.array([detune_slow_factor])
        tempinput['R'] = np.array([R])
        tempinput['GMTSQ'] = np.array([GMTSQ])
        tempinput['Gamma'] = np.array([Gamma])
        tempinput['nBeam'] = np.array([nBeam])
        tempinput['beam_shift'] = np.array([beam_shift])
        
        tempinput['n_turns'] = np.array([n_turns])
        tempinput['n_dynamicOn'] = np.array([n_dynamicOn])
        tempinput['n_bunches'] = np.array([n_bunches])
        tempinput['n_fill'] = np.array([n_fill])
        tempinput['n_q_ramp'] = np.array([n_q_ramp])
        tempinput['n_detune_start'] = np.array([n_detune_start])
        tempinput['n_detune_ramp'] = np.array([n_detune_ramp])
        tempinput['n_detune_ramp_tot'] = np.array([n_detune_ramp_tot])
        tempinput['n_I_ramp_start'] = np.array([n_I_ramp_start])
        tempinput['n_I_ramp_end'] = np.array([n_I_ramp_end])
        tempinput['step_store'] = np.array([step_store])
        tempinput['Prad'] = np.array([Prad])
        tempinput['t_rad_long'] = np.array([t_rad_long])
        tempinput['Ek_damp'] = np.array([Ek_damp])
        tempinput['Npar'] = np.array([Npar])
        tempinput['NperBunch'] = np.array([nPar])
        tempinput['N_bins'] = np.array([N_bins])
        tempinput['fill_step'] = np.array([fill_step])
        tempinput['siglong'] = np.array([siglong])
        tempinput['A'] = np.array([A])
        
        tempinput['nRF'] = np.array([nRF])
        tempinput['nRF1'] = np.array([nRF1])
        tempinput['nRFc'] = np.array([nRFc])
        tempinput['nRF2'] = np.array([nRF2])
        tempinput['nHOM'] = np.array([nHOM])
        tempinput['nCav'] = np.array(nCav)
        tempinput['h'] = np.array(h)
        tempinput['I_Q_ref_ini'] = np.array(I_Q_ref_ini)
        tempinput['I_Q_ref_final'] = np.array(I_Q_ref_final)
        tempinput['RoQ'] = np.array(RoQ)*nCav
        tempinput['delay'] = np.array(delay)
        tempinput['n_fb_on'] = np.array(n_fb_on)
        tempinput['gII'] = np.array(gII)
        tempinput['gQQ'] = np.array(gQQ)
        tempinput['gIQ'] = np.array(gIQ)
        tempinput['gQI'] = np.array(gQI)
        tempinput['gIIi'] = np.array(gIIi)
        tempinput['gQQi'] = np.array(gQQi)
        tempinput['gIQi'] = np.array(gIQi)
        tempinput['gQIi'] = np.array(gQIi)
        tempinput['PA_cap'] = np.array(PA_cap)
        
        tempinput['QL'] = np.array(QL)
        tempinput['Vref_I'] = np.array(VrefI)*nCav
        tempinput['Vref_Q'] = np.array(VrefQ)*nCav
        tempinput['Iref_I'] = np.array(I_I)
        tempinput['Iref_Q'] = np.array(I_Q)
        tempinput['I_I_ref_ini'] = np.array(I_I_ini)
        tempinput['I_I_ref_final'] = np.array(I_I)
        tempinput['I_Q_ref_ini'] = np.array(I_Q_ini)
        tempinput['I_Q_ref_final'] = np.array(I_Q)
        tempinput['detune'] = np.array(detune)
        tempinput['detune_ini'] = np.array(delta_f_ini)
        tempinput['detune_mid'] = np.array((delta_f-delta_f_ini)/2)
        tempinput['detune_final'] = np.array(delta_f)
        

        fn1 = 'input.txt'
        inputfile1 = os.path.join(cwd,fn1)
        with open(inputfile1,'w') as wrt_to_input:
            for i in tempinput:
                wrt_to_input.write(str(i)+' ')
                for j in range(len(tempinput[i])):
                    wrt_to_input.write(str(tempinput[i][j])+' ')
                    #print(tempinput[i][j])
                wrt_to_input.write('\n')
        print("Generated input file.")
# please leave the one that is going be used uncommented, and comment out the other two.

        #args = ("../APES")
        #args = ("../APESAVX2")
        #args = ("../APESGCC")
        args = ("../../runavx2.sh")
        print(cwd)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE,cwd=cwd)
        print("Simulation started...")
        err = popen.wait()
        output = popen.stdout.read()
        print(output.decode("utf-8"))


        path = os.path.join(cwd,"{0:02d}".format(charge_factor)+"{0:02d}".format(thetaL_factor)+"nmacro{0:.0f}".format(Npar)+"_nBin{0:.0f}".format(N_bins)+"_Idc{0:.2f}A".format(n_bunches*nPar*f0*1.6e-19)+"_ThetaL{0:.1f}degree".format(180/pi*thetaL[0]))
        
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s" % path)
        files = os.listdir(cwd)
        result_fn = [i for i in files if i[-3:]=='bin' and i!='par.bin']
        for i in result_fn:
            path_result_fn = os.path.join(cwd,i)
            subprocess.call(["mv",path_result_fn,path])
        path_out = os.path.join(cwd,"out")
        subprocess.call(["cp",path_out,path])

        path_in = os.path.join(cwd,"input.txt")
        subprocess.call(["cp",path_in,path])
        # convert the RoQ back to per cavity, otherwise it's wrong
        RoQ = RoQ*nCav
os.chdir(home)
