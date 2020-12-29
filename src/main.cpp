#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cmath>
#include <string.h>
#include <sstream>
#include <iterator>
#include <map>
#include <cstdlib>
//#include <aligned_new>
#include <mkl.h>
#include <mkl_vsl.h>
//#include <mpi.h>

#include <parallel/algorithm>
#include <parallel/numeric>

#include "const.h"
#include "inputPara.h"


double qovermp = qe/mp;
double qoverme = qe/me;
double qovermAu = qAu/mAu;
int qovermp_on = 1;
int qoverme_on = 1;
int count = 0;
const int OMP_NUM_THREADS = omp_get_max_threads()/2;

void getIQ(double* data, double* sintable, double* costable, double* tempI, double* tempQ, double* I, double* Q, 
            int delay, int nSmp, 
            int i, int nRF, int stride,int nBeam){
    int j = (i-delay-1)%nSmp;
    double nSmpInver = 1.0/nSmp;
#pragma vector always
#pragma ivdep
    for(int rf = 0;rf<nRF;++rf){
        int vidx = (i-delay)*stride+1+2*nBeam+3*rf;
        I[rf] = I[rf] - (tempI[j*nRF+rf]-data[vidx]*sintable[i*nRF+rf])*nSmpInver*2;
        Q[rf] = Q[rf] - (tempQ[j*nRF+rf]-data[vidx]*costable[i*nRF+rf])*nSmpInver*2;
        tempI[j*nRF+rf] = data[vidx]*sintable[i*nRF+rf];
        tempQ[j*nRF+rf] = data[vidx]*costable[i*nRF+rf];
    }
}

void updateVI(double *data,
                double* k, 
                int nData_tot,
                double dt, 
                double* I_I_ref,
                double* I_Q_ref,
                double* V_I_ref,
                double* V_Q_ref,
                double* PA_cap,
                double* R,
                double* gii,
                double* gqq,
                double* I,
                double* Q,
                double* sintable, 
                double* costable, 
                double* tempI,
                double* tempQ,
                std::vector<std::vector<double>>& Sp,
                std::vector<std::vector<double>>& S,
                std::vector<double> epsilon_comb,
                std::vector<double> g_comb,
                int* delay, 
                int* nSmp,
                int nRF, int nRF1,int nRFc, int nRF2,int stride,int nBeam){ // k is the constant parameter vector
    double* Rinv = new double[nRF];
    int iprv = 0;
    for(int i = 0;i<nRF;++i){
        Rinv[i] = 1/R[i];
    }
    for(int i = delay[0]+nSmp[0];i<nData_tot;++i){
        //first calculate the previous index
        iprv = i-1;
        // then update the voltage
        // the meanning of the terms are V(i-1), Ig(i-1), Ib(i), Ib(i-1),U(i-1)
#pragma ivdep       
        for(int j = 0;j<nRF1;++j){
            //data[i*stride+1+2*nBeam+j*3] = data[iprv*stride+1+2*nBeam+j*3]*k[j*5]+(data[iprv*stride+1+2*nBeam+j*3+1]+data[i*stride+1+2*nBeam+j*3+1])/2*k[j*5+1]+data[iprv*stride+1]*k[j*5+2]+data[iprv*stride+2]*k[j*5+3]+data[iprv*stride+1+2*nBeam+j*3+2]*k[j*5+4];
            data[i*stride+1+2*nBeam+j*3] = data[iprv*stride+1+2*nBeam+j*3]*k[j*5]+(data[iprv*stride+1+2*nBeam+j*3+1]*0.5+data[i*stride+1+2*nBeam+j*3+1]*0.5)*k[j*5+1]+data[i*stride+1]*k[j*5+2]+data[i*stride+2]*k[j*5+3]+data[iprv*stride+1+2*nBeam+j*3+2]*k[j*5+4];
            data[i*stride+1+2*nBeam+j*3+2] = data[iprv*stride+1+2*nBeam+j*3+2]+dt*data[i*stride+1+2*nBeam+j*3];
        }
#pragma ivdep
        for(int j = nRF1;j<nRF1+nRFc;++j){
            data[i*stride+1+2*nBeam+j*3] = data[iprv*stride+1+2*nBeam+j*3]*k[j*5]+data[iprv*stride+1+2*nBeam+j*3+1]*k[j*5+1]+(data[i*stride+1]+data[i*stride+3])*k[j*5+2]+(data[i*stride+2]+data[i*stride+4])*k[j*5+3]+data[iprv*stride+1+2*nBeam+j*3+2]*k[j*5+4];
            data[i*stride+1+2*nBeam+j*3+2] = data[iprv*stride+1+2*nBeam+j*3+2]+dt*data[i*stride+1+2*nBeam+j*3];
        }
#pragma ivdep
        for(int j = nRF1+nRFc;j<nRF1+nRFc+nRF2;++j){
            data[i*stride+1+2*nBeam+j*3] = data[iprv*stride+1+2*nBeam+j*3]*k[j*5]+data[iprv*stride+1+2*nBeam+j*3+1]*k[j*5+1]+data[i*stride+3]*k[j*5+2]+data[i*stride+4]*k[j*5+3]+data[iprv*stride+1+2*nBeam+j*3+2]*k[j*5+4];
            data[i*stride+1+2*nBeam+j*3+2] = data[iprv*stride+1+2*nBeam+j*3+2]+dt*data[i*stride+1+2*nBeam+j*3];
        }
        // then we need to get the I, Q to update current
        getIQ(data,sintable, costable, tempI, tempQ, I,Q,delay[0],nSmp[0],i,nRF,stride,nBeam);

        for(int j = 0; j<nRF; ++j){
            S[j][i-nSmp[0]] = (1-epsilon_comb[j])*Sp[j][i-nSmp[0]]+epsilon_comb[j]*data[i*stride+1+2*nBeam+j*3];
            Sp[j][i-nSmp[0]] = S[j][i-nSmp[0]];
        }

        // update Ig for all frequency. 
#pragma ivdep
        for(int j = 0;j<nRF;++j){
            // direct IQ feedback
            data[i*stride+1+2*nBeam+j*3+1] = (I_I_ref[j]+(V_I_ref[j]-I[j])*Rinv[j]*gii[j])*sintable[i*nRF+j]+(I_Q_ref[j]+(V_Q_ref[j]-Q[j])*Rinv[j]*gqq[j])*costable[i*nRF+j];
            // comb
            //data[i*stride+1+2*nBeam+j*3+1] += g_comb[j]*Rinv[j]*(V_I_ref[j]*sintable[i*nRF+j]+V_Q_ref[j]*costable[i*nRF+j]-S[j][i-nSmp[0]-delay[0]]);
            data[i*stride+1+2*nBeam+j*3+1] += g_comb[j]*(I_I_ref[j]*sintable[i*nRF+j]+I_Q_ref[j]*costable[i*nRF+j]-Rinv[j]*(S[j][i-nSmp[0]-delay[0]]));
            //data[i*stride+1+2*nBeam+j*3+1] += g_comb[j]*Rinv[j]*S[j][i-nSmp[0]-delay[0]];

        }
    }
    delete[] Rinv;    
}

void update_index_neighbor(double* bunch, double* weight, double dt, double Trev, int fill_count, 
                            int nPar, int totBin, int shift, int nBunch,int nBin,int shift_beam,int nBucket,
                            int* index, int* neighbor, int* parFlag){
    double dtInver = 1/dt;
    int totN = fill_count*nPar;
    // first totN data in bunch array is 'time' info of the particles.
#pragma omp parallel for schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep
    for(int i = 0;i<totN;++i){
        // first locate the particle on time grid. becareful that the particle time coordinates can be out of the bound
        //index[i] = (bunch[i] - Trev*fmod(bunch[i],Trev))*dtInver;
        index[i] = (totBin+int((fmod(bunch[i],Trev))*dtInver))%totBin;
#if 1
        if(index[i]<0|index[i]>totBin){
            //std::cout<<"Particle "<<i<<" is lost. "<<std::endl;
            parFlag[i] = 0;
        }
#endif
        // the we need to decide the weight of the particle on the grid

        weight[i] = fmod(bunch[i],Trev)-(dt*index[i]+dt*0.5);
        //std::cout<<weight[i]<<','<<dt<<','<<bunch[i]<<','<<Trev<<','<<fmod(bunch[i],Trev)<<','<<(dt*index[i]+dt*0.5)<<','<<index[i]<<std::endl;;
        int x = weight[i]>0?1:-1;
        neighbor[i] = (index[i] + x)%totBin;
        index[i] += shift;
        neighbor[i] += shift;
    }
}

void update_M1(double* particles, double* M1, int nPar, int fill_count,int* Npar_survive,int* parFlag){
    double nParInver = 1.0/nPar;
    for(int i = 0;i<fill_count;++i){
        Npar_survive[i] = std::accumulate(parFlag+i*nPar,parFlag+(i+1)*nPar,0.0);
    }
    double* temppar = new double[nPar*fill_count*2];
    for (int i = 0;i<fill_count*nPar;++i){
        temppar[i] = particles[i]*parFlag[i];
        temppar[fill_count*nPar+i] = particles[fill_count*nPar+i]*parFlag[i];
    }
    for(int i = 0;i<fill_count;++i){
        M1[i] = std::accumulate(temppar+i*nPar,temppar+(i+1)*nPar,0.0)/Npar_survive[i];
        M1[fill_count+i] = std::accumulate(temppar+(fill_count+i)*nPar,temppar+(fill_count+i+1)*nPar,0.0)/Npar_survive[i];
    }
    delete[] temppar;
}

void update_M1_0(double* particles, double* M1, int nPar, int fill_count,int* Npar_survive,int* parFlag){
    double nParInver = 1.0/nPar;
    for(int i = 0;i<fill_count;++i){
        Npar_survive[i] = std::accumulate(parFlag+i*nPar,parFlag+(i+1)*nPar,0.0);
    }
    
    for(int i = 0;i<fill_count;++i){
        M1[i] = std::accumulate(particles+i*nPar,particles+(i+1)*nPar,0.0)*nParInver;
        M1[fill_count+i] = std::accumulate(particles+(fill_count+i)*nPar,particles+(fill_count+i+1)*nPar,0.0)*nParInver;
    }
}

void update_M2(double* particles, double* M1, double* M2, int nPar, int fill_count,int* Npar_survive,int* parFlag){
    // M2 store the sig_t_sq, sig_gamma_sq and sig_t_sig_gamma
    double nParInver = 1.0/nPar;
#pragma omp parallel for schedule(static,nPar) num_threads(OMP_NUM_THREADS)
    for(int i = 0;i<fill_count;++i){
        for(int j = 0;j<nPar;++j){
            M2[i] += (particles[i*nPar+j]-M1[i])*(particles[i*nPar+j]-M1[i])*parFlag[i*nPar+j];//time
            M2[fill_count+i] += (particles[fill_count*nPar+i*nPar+j]-M1[fill_count+i])*(particles[fill_count*nPar+i*nPar+j]-M1[fill_count+i])*parFlag[i*nPar+j]; // gamma
            M2[fill_count*2+i] += (particles[i*nPar+j]-M1[i])*(particles[fill_count*nPar+i*nPar+j]-M1[fill_count+i])*parFlag[i*nPar+j];//cross correlation
        }
        M2[i] = M2[i]/Npar_survive[i];
        M2[fill_count+i] = M2[fill_count+i]/Npar_survive[i];
        M2[fill_count*2+i] = M2[fill_count*2+i]/Npar_survive[i];
    }
    
}

void updateParticle(int type_of_particle, double* particle_coords, double* data, double* weight, double* M1,
                    double Gamma0, double eta, double dt, double Trev, double Ek_damp,double Urad0,
                    int h,int fill_step,int fill_count, int nPar, int nRF, int nHOM,int nBeam,int nRF1, int nRFc, int nRF2,int stride,double* nCav,
                    int* index, int* neighbor, int* parFlag){

    int nTot = fill_count*nPar;
    //int stride = 3+3*nRF+2*nHOM; // time, Ib[i-1], Ib[i], V0, Ig0, U0,
    double dtInver = 1/dt;
    double Gamma0Inver = 1/Gamma0;
    double nParInv = 1/nPar; 
    double qoverm;
    switch(type_of_particle)
    {
        case 0:
            qoverm = qovermp;
            break;
        case 2:
            qoverm = qovermAu;
            break;
        default:
            qoverm = qovermp;
    }
    // gamma of first beam
    int j = 0;
    for(int j = 0;j<nRF1+nRFc;++j){
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep
        for(int i = 0;i<nTot/nBeam;++i){
            int par_idx = index[i];
            int par_neighbor = neighbor[i];
            double V = 0;
            V = data[par_idx*stride+1+2*nBeam+j*3]+(data[par_neighbor*stride+1+2*nBeam+j*3]-data[par_idx*stride+1+2*nBeam+j*3])*dtInver*fabs(weight[i]);
            particle_coords[nTot+i] += qoverm*(V-Urad0*nCav[j])*c_inver*c_inver*qovermp_on;// Urad0 represent the unaccounted HOM power
        }
    }

    // gamma of second beam
    for(int j = nRF1;j<nRF1+nRFc+nRF2;++j){
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep
        for(int i = nTot/nBeam;i<nTot;++i){
            int par_idx = index[i];
            int par_neighbor = neighbor[i];
            double V = 0;
            V = data[par_idx*stride+1+2*nBeam+j*3]+(data[par_neighbor*stride+1+2*nBeam+j*3]-data[par_idx*stride+1+2*nBeam+j*3])*dtInver*fabs(weight[i]);
            particle_coords[nTot+i] = particle_coords[nTot+i]-qoverm*(V-Urad0*nCav[j])*c_inver*c_inver*qovermp_on; // '-' sign in front of qovermp means this is going in opposite direction.
        }
    }

    // artificial dampping
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep 
    for(int i = 0;i<nTot;++i){
        particle_coords[nTot+i] -= (M1[fill_count+int(i*nParInv)]-Gamma0)*Ek_damp*qovermp_on; // damping term
    }

    // t
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS) 
    for(int i = 0;i<nTot;++i){
        int iBunch = i/nPar;
        double tstart = Trev/h*iBunch*fill_step;
        double tend = Trev/h*(iBunch*fill_step+1);
        particle_coords[i] += Trev/(1/(eta*(particle_coords[nTot+i]*Gamma0Inver-1))-1)*qovermp_on;
        if(particle_coords[i]<tstart | particle_coords[i]>tend){
            //std::cout<<particle_coords[i]<<','<<tstart<<','<<tend<<std::endl;
            //std::cout<<i<<','<<iBunch<<std::endl;
            //exit(-1);
            parFlag[i] = 0;
        }
        //particle_coords[i] = fmod(particle_coords[i]+Trev,Trev);
    }
    
    //update_M1(particle_coords, M1, nPar, fill_count);
}
void updateParticle_e(double* particle_coords, double* data, double* weight, double* M1,double* rand_table,
                    double Gamma0, double eta, double dt, double Trev, double Ek_damp,double Urad0,double coeff_decay_long, double coeff_excite_long,
                    int fill_count, int nPar, int nRF, int nHOM,int nBeam,int nRF1, int nRFc, int nRF2,int stride,double* nCav,
                    int* index, int* neighbor){

    int nTot = fill_count*nPar;
    //int stride = 3+3*nRF+2*nHOM; // time, Ib[i-1], Ib[i], V0, Ig0, U0,
    double dtInver = 1/dt;
    double Gamma0Inver = 1/Gamma0;
    double nParInv = 1/nPar; 
    
    // gamma of first beam
    for(int j = 0;j<nRF1+nRFc;++j){
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep
        for(int i = 0;i<nTot/nBeam;++i){
            int par_idx = index[i];
            int par_neighbor = neighbor[i];
            //std::cout<<par_idx<<','<<par_neighbor<<std::endl;
            double V = 0;
            V = data[par_idx*stride+1+2*nBeam+j*3]+(data[par_neighbor*stride+1+2*nBeam+j*3]-data[par_idx*stride+1+2*nBeam+j*3])*dtInver*fabs(weight[i]);
            particle_coords[nTot+i] += qoverme*(V-Urad0*nCav[j])*c_inver*c_inver*qoverme_on;//radiation
        }
    }

    // artificial dampping
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep 
    for(int i = 0;i<nTot;++i){
        particle_coords[nTot+i] -= (M1[fill_count+int(i*nParInv)]-Gamma0)*Ek_damp*qoverme_on; // damping term
       
    }

// minic the sychrotron radiation dampping and quantum excitation 
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS)
#pragma ivdep 
    for(int i = 0;i<nTot;++i){
        particle_coords[nTot+i] -= qoverme_on*(coeff_decay_long*(particle_coords[nTot+i]-Gamma0)+coeff_excite_long*(2*double(rand_table[i])-1));
    }

    // tQs0
#pragma omp parallel for //schedule(static,nPar) num_threads(OMP_NUM_THREADS) 
    for(int i = 0;i<nTot;++i){
        particle_coords[i] += Trev/(1/(eta*(particle_coords[nTot+i]*Gamma0Inver-1))-1)*qoverme_on;
        particle_coords[i] = fmod(particle_coords[i]+Trev,Trev);
    }
}



void updateIb(double* particle_coords, double* data, double* weight, 
            double dt, double* q, 
            int fill_count, int nPar, int totBin, int shift, int stride,int nBeam,
            int* index, int* neighbor,int* parFlag){
    double dtInver = 1/dt;
    int nBunchPerBeam = int(fill_count/nBeam);
    // first reset the Ibs

    for(int j = 0;j<nBeam;++j){
#pragma vector always
#pragma ivdep
        for(int i = 0; i<totBin; ++i){
            data[i*stride+j*2+1] = 0;
            data[i*stride+j*2+2] = 0;
        }
    }

    for(int j = 0;j<nBeam;++j){
        for(int i = j*nBunchPerBeam*nPar; i<(j+1)*nBunchPerBeam*nPar ; ++i){
            if(i/nPar==0){
                //std::cout<<index[i]<<std::endl;
            }
            data[(index[i])*stride+j*2+1] += q[j]*(1-fabs(weight[i]*dtInver))*dtInver*parFlag[i];
            data[(neighbor[i])*stride+j*2+1] += q[j]*fabs(weight[i]*dtInver)*dtInver*parFlag[i];
            data[(index[i]+1)*stride+j*2+2] += q[j]*(1-fabs(weight[i]*dtInver))*dtInver*parFlag[i];
            data[(neighbor[i]+1)*stride+j*2+2] += q[j]*fabs(weight[i]*dtInver)*dtInver*parFlag[i];
        }
    }
}

void init_bunch(double* bunch,double t0,double gamma0,double sigt,double siggamma,int Npar, int nBunch,int bunch_idx,int nBeam){
    /* So far only the Gaussian distribution is supported */
    std::default_random_engine generator(1); // Normal distribution generator
    generator.seed(0);
    std::normal_distribution<double> dist0(t0, sigt); // normal distribution in t
    std::normal_distribution<double> dist1(0, siggamma); // normal distribution in gamma    
    
    for(int i = 0;i<Npar;++i){
        bunch[bunch_idx*Npar+i] = dist0(generator);
        
        while(abs(bunch[bunch_idx*Npar+i]-t0)>3*sigt){
            //std::cout<<"re_throw"<<std::endl;
            bunch[bunch_idx*Npar+i] = dist0(generator);
        }
        
    }
    for(int i = 0 ;i<Npar;++i){
        bunch[nBunch*nBeam*Npar+bunch_idx*Npar+i] = (dist1(generator)+1)*gamma0;//0.0*cos(2*pi*bunch_idx/120)+gamma0;
        
        while(abs(bunch[nBunch*nBeam*Npar+bunch_idx*Npar+i]-gamma0)>3*siggamma*gamma0){
            bunch[nBunch*nBeam*Npar+bunch_idx*Npar+i] = (dist1(generator)+1)*gamma0;
        }
        
    }
    bunch[bunch_idx*Npar] = t0; // set the first particle to the reference particle.
    bunch[nBunch*nBeam*Npar+bunch_idx*Npar] = gamma0;
    //bunch[nBunch*Npar+bunch_idx] = gamma0+1;
}

void init_tempIQ(double* data, double* sintable, double* costable, double* tempI, double* tempQ, 
                double* I, double* Q, 
                int* delay, int* nSmp, int i,int nRF,int stride){
// initialize the temp arrays that hold the intermediate results for I Q calculation.    
    for(int j = 0;j<nRF;++j){    
        int idx_temp = 0;
        for(int idx = i-delay[0]-nSmp[0]+1;idx<i-delay[0]+1;++idx){
        tempI[idx_temp*nRF+j] = data[idx*stride+3+j*3]*sintable[idx*nRF+j];
        tempQ[idx_temp*nRF+j] = data[idx*stride+3+j*3]*costable[idx*nRF+j];
        idx_temp++;
        }
        for(int idx = 0;idx<nSmp[0];++idx){
            I[j] += tempI[idx*nRF+j];
            Q[j] += tempQ[idx*nRF+j];
        }
        I[j] = I[j]/nSmp[0]*2;
        Q[j] = Q[j]/nSmp[0]*2;
    }
}

void current_rampping(int turn, int n_I_ramp_start, int n_I_ramp_end, int nRF, double dt, 
                        double* I_I_ref, double* I_Q_ref, double* I_I_ref_ini, double* I_I_ref_final, double* I_Q_ref_ini, double* I_Q_ref_final){
    double rampInv = 1.0/double(n_I_ramp_end-n_I_ramp_start);
    #pragma ivdep 
    for(int i = 0;i<nRF;++i){
        I_I_ref[i] = (turn-n_I_ramp_start)*(I_I_ref_final[i]-I_I_ref_ini[i])*rampInv+I_I_ref_ini[i];
        I_Q_ref[i] = (turn-n_I_ramp_start)*(I_Q_ref_final[i]-I_Q_ref_ini[i])*rampInv+I_Q_ref_ini[i];
    }
}

void detune_rampping(int turn, int n_detune_start, int n_detune_ramp,int nRF, double dt,
            double* omegac, double* omegarf, double* QL, double* C, double* R, double* Leff,double* k,
            double* detune, double* detune_ini, double* detune_final){
    double rampInv = 1.0/double(n_detune_ramp-n_detune_start);
#pragma ivdep 
    for(int i = 0;i<nRF;++i){
        detune[i] = (turn-n_detune_start)*(detune_final[i]-detune_ini[i])*rampInv+detune_ini[i];
        omegac[i] = omegarf[i]+detune[i]*2*pi;
        C[i] = QL[i]/R[i]/omegac[i];//[2] = {QL[0]/R[0]/omegac[0],QL[1]/R[1]/omegac[1]};//9.2755373928124199e-11;
        Leff[i] = dt*dt/(1-cos(dt*omegac[i]))/2/C[i];//1/(C[i]*omegac[i]*omegac[i]);//

        k[i*5] = 1-dt/R[i]/C[i];
        k[i*5+1] = dt/C[i];
        k[i*5+2] = -dt/C[i]/2;
        k[i*5+3] = k[i*5+2];
        k[i*5+4] = -dt/C[i]/Leff[i];
    }
}

void get_init_t0s(double* t0s,double dT, double t0, double Trev, int nBeam, int nBunch, int shift,int nBucket,std::vector<int> &mode, std::vector<double> &amp,int n_ini_CBI = 0){
// get the initial time coordinates of each bunches stored in array "t0s", dT is the bunch distance, t0 is the start t
// nBeam is the number of beams, can be one or two
// nBunch is the number of bunches in each beam
// shift is the relative shift between two bunches 
// 'mode' is the mu number for coupled bunch mode initially excited
// 'amp' is the amplitude of the coupled bunch mode, can excite only one mode now.
    std::cout<<"shift is : "<<shift<<std::endl;
    for(int i = 0;i<nBeam;++i){
        for(int j = 0;j<nBunch;++j){
            t0s[i*nBunch+j] = fmod(t0+(i*shift)%nBucket*t0*2+j*dT+i*t0/2,Trev);
            // apply initial coupled bunch mode if any
            for(int k = 0;k<n_ini_CBI;++k){
                t0s[i*nBunch+j] += amp[k]*sin(mode[k]*2*pi*1/Trev*t0s[i*nBunch+j]);
            }
            
        }

        std::cout<<"Time of first bunch: "<<t0s[i*nBunch+0]<<std::endl;
    }
}

void get_init_gamma0s(double* gamma0s,double* t0s,double dT, double t0, double Trev, double Gamma0, int nBeam, int nBunch, int shift,int nBucket,std::vector<int> &mode, std::vector<double> &amp,int n_ini_CBI = 0){
// get the initial gamma coordinates of each bunches stored in array "gamma0s", dT is the bunch distance, t0 is the start t
// nBeam is the number of beams, can be one or two
// nBunch is the number of bunches in each beam
// shift is the relative shift between two bunches 
    std::cout<<"shift is : "<<shift<<std::endl;
    for(int i = 0;i<nBeam;++i){
        for(int j = 0;j<nBunch;++j){
            gamma0s[i*nBunch+j] = Gamma0;
            for(int k = 0;k<n_ini_CBI;++k){
                gamma0s[i*nBunch+j] += amp[k]*cos(mode[k]*2*pi/Trev*t0s[i*nBunch+j]);
            }
            
        }
    }
}
// generate the random number table for the quantum excitation calculation each turn. 
// this one is for gcc compiler, if intel compiler is not available 
void get_rand_table(double* rand_table, int nBunch, int nPar){
    for(int i = 0;i<nBunch*nPar;++i){
        rand_table[i] = double(rand())/double(RAND_MAX);
    }
}

#if 0
// using the stream way to generate random numbers, faster than the one above. 
void get_rand_table1(double* rand_table, int nBunch, int nPar,int turn){
    VSLStreamStatePtr stream;
    auto status = vslNewStream(&stream, VSL_BRNG_SFMT19937, turn);
    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
		stream, nBunch*nPar, rand_table, 0, 1);
}
#endif 

int main(){
    std::cout.precision(17);
    omp_set_num_threads(OMP_NUM_THREADS);

    inputPara input1;
    input1.read("input.txt");    
    input1.printout();
    int type_of_particle = input1.generalPara["type"];
    int csv_out = int(input1.generalPara["csv_out"]);
    int dynamicOn = input1.generalPara["dynamicOn"];
    int n_dynamicOn = input1.generalPara["n_dynamicOn"];
    int n_Ek_damp_on = input1.generalPara["n_Ek_damp_on"];// typically set to 0
    int n_Ek_damp_off = input1.generalPara["n_Ek_damp_off"];

    int N = input1.generalPara["n_turns"];
    int step = input1.generalPara["step_store"];
    int npRF = input1.bunchPara["N_bins"];
    int nRF1 = input1.ringPara["nRF1"];
    int nRF2 = input1.ringPara["nRF2"];
    int nRFc = input1.ringPara["nRFc"];    
    int nRF = nRF1+nRFc+nRF2;

    int nHOM = input1.ringPara["nHOM"];// not realy in use.
    int nBunch = input1.generalPara["n_bunches"]; 
    int nBeam = input1.bunchPara["nBeam"];
    int beam_shift = input1.bunchPara["beam_shift"]; // the shift between two beams in terms of fundamental bucket
    int n_q_ramp = input1.generalPara["n_q_ramp"]; // the number of turns it takes to ramp up the charge.
    
    int n_detune_start = input1.generalPara["n_detune_start"]; // the turn we start the detuning
    int n_detune_ramp = input1.generalPara["n_detune_ramp"]; // the turn we switch the tuning spped
    int detune_slow_factor = input1.generalPara["detune_slow_factor"]; // some factor I used to slow down the detune ramping for 56 MHz, now this is set to 1.
    int n_detune_ramp_tot = input1.generalPara["n_detune_ramp_tot"];// the turn we stop tuning.
    int n_I_ramp_start = input1.generalPara["n_I_ramp_start"];// the turn we start to ramp the current, usually from zeroth turn
    int n_I_ramp_end = input1.generalPara["n_I_ramp_end"];// the turn we end the current rampping. 
    int n_fill = input1.generalPara["n_fill"]; // the turn number when we start filling.
    // for simplicity, setting the n_Ek_damp_off to n_fill+n_q_ramp
    n_Ek_damp_off = n_fill+n_q_ramp;
    int fill_step = input1.bunchPara["fill_step"]; // fill pattern
    int fill_count = 0; // the numbe of bunches that is really in the ring.
    int nPar = input1.bunchPara["Npar"]; // number of macro particles
    int atomicZ = 1;
    double NperBunch = input1.bunchPara["NperBunch"];  // number of real particles in one bunch. 
    double* q = new double[nBeam]; // charge per bunch, use this to ramp the charge
    double QperBunch; // charge per bunch
    double Ek; // energy of the reference particle for the ring bunch. 
    double Gamma0 = input1.ringPara["Gamma"]; 
    

    switch(type_of_particle){
            case 0: // proton
                atomicZ = 1;
                Ek = Gamma0*E0p;
                E0 = E0p;
                QperBunch = NperBunch*qe;
                break;
            case 1: // electron
                atomicZ = 1;
                Ek = Gamma0*E0e;
                QperBunch = NperBunch*qe;
                E0 = E0e;
                break;
            case 2: // Au
                atomicZ = 79;
                Ek = Gamma0*E0Au;
                QperBunch = NperBunch*qe*79;
                E0 = E0Au;
                break;
        }
    double q0 = QperBunch/nPar; // target charge per bunch
    double dqodn = QperBunch/nPar/n_q_ramp; // ramp speed of the charge

    double Rring = input1.ringPara["R"]; // radius of the ring
    double GMTSQ = input1.ringPara["GMTSQ"]; // transition gamma square. 
    double A = input1.bunchPara["A"]; // [eV.s]
    double t_rad_long = input1.ringPara["t_rad_long"]; // longitudinal radiation damping time
    double siglong = input1.bunchPara["siglong"]; // rms spread in gamma
    double epsilon = A/6;
   
    int* index = new int[nPar*nBunch*nBeam]; // store the index of particles, the index indicates the time grid
    int* neighbor = new int[nPar*nBunch*nBeam]; //store the index of the neighbor of the particle
    double* bunch = new double[nPar*2*nBunch*nBeam];// to store coordinates of the bunch in time and gamma.
    double *temp = new double[nPar*nBunch*nBeam*2];
    int* parFlag = new int[nPar*nBunch*nBeam]; // the functionality is not implemented yet, flag that marks the partcle: 1 means under tracking, 0 means lost
    int* nPar_survive = new int[nBunch*nBeam]; // not in use yet
    double* M1 = new double[nBunch*2*nBeam]; // to store the centroids of each bunch. 
    double* M2 = new double[nBunch*3*nBeam]; // to store the sigma**2s' of each bunch. 
    double* weight = new double[nPar*nBunch*nBeam]; // the weigth of one particle on its own bin(and its neighoring bin is 1-weight)
    double* first_archive = new double[(N-n_fill)*2]; // store the first bunch for all turns
    double omega0 = 2*pi*c_light*sqrt(1-1/Gamma0/Gamma0)/(2*pi*Rring);
    double Trev = 2*pi/omega0;
    double Prad = input1.ringPara["Prad"];
    double Urad0 = Prad/(QperBunch*nBunch/Trev); // voltage corresponding to the radiation loss. 

    int* h = new int[nRF1+nRFc+nRF2];
    int* delay = new int[nRF1+nRFc+nRF2]; // using same delay for all the RF's now, in unit of dt, the smallest time sample
    int* nSmp = new int[nRF1+nRFc+nRF2]; // number of samples used to calculate I and Q, default is npRF
    int* fb_on = new int[nRF1+nRFc+nRF2]; // the turn number when we turn on the feedback
    double* Ig_I_ref = new double[nRF1+nRFc+nRF2];
    double* Ig_Q_ref = new double[nRF1+nRFc+nRF2];

    double* I_I_ref_ini = new double[nRF1+nRFc+nRF2];
    double* I_I_ref_final = new double[nRF1+nRFc+nRF2];    
    double* I_Q_ref_ini = new double[nRF1+nRFc+nRF2];
    double* I_Q_ref_final = new double[nRF1+nRFc+nRF2];
    double* V_I_ref = new double[nRF1+nRFc+nRF2];
    double* V_Q_ref = new double[nRF1+nRFc+nRF2];
    double* gii = new double[nRF1+nRFc+nRF2];
    double* gqq = new double[nRF1+nRFc+nRF2];
    double* gii_rec = new double[nRF1+nRFc+nRF2];
    double* gqq_rec = new double[nRF1+nRFc+nRF2];
    double* I = new double[nRF1+nRFc+nRF2];
    double* Q = new double[nRF1+nRFc+nRF2];
    
    double* omegarf = new double[nRF1+nRFc+nRF2];    
    double* detune = new double[nRF1+nRFc+nRF2];
    double* detune_ini = new double[nRF1+nRFc+nRF2];
    double* detune_mid = new double[nRF1+nRFc+nRF2];
    double* detune_final = new double[nRF1+nRFc+nRF2];
    double* PA_cap = new double[nRF1+nRFc+nRF2];// can be used to cap the Ig, not implemented yet.
    double* nCav = new double[nRF1+nRFc+nRF2];
    int main_detune = input1.generalPara["main_detune"]; // this is used to calculate the tuning speed.
    double* omegac = new double[nRF1+nRFc+nRF2]; // omega of cavities
    double* QL = new double[nRF1+nRFc+nRF2]; // Loaded Q
    double* RoQ = new double[nRF1+nRFc+nRF2];
    double* R = new double[nRF1+nRFc+nRF2];
    double* C = new double[nRF1+nRFc+nRF2];//
    double* Leff = new double[nRF1+nRFc+nRF2];//
    int* on = new int[nRF1+nRFc+nRF2];
    double dt = 0;
    int stride = 1+2*nBeam+3*(nRF1+nRFc+nRF2); //stride in the array that stores the VI datas: time, 2 Ibs for each beam, V, Ig, U for each RF
    std::cout<<"test1"<<std::endl;
    for (int i = 0;i<nRF1+nRFc+nRF2;++i){

        h[i] = input1.rfPara["h"][i];
        delay[i] = input1.apPara["delay"][i];
        nSmp[i] = npRF;
        fb_on[i] = input1.apPara["n_fb_on"][i];
        omegarf[i] = omega0*h[i];    
        dt = 2*pi/omegarf[0]/npRF;
        detune[i] = input1.apPara["detune"][i];
        detune_ini[i] = input1.apPara["detune_ini"][i];
        detune_mid[i] = input1.apPara["detune_mid"][i];
        detune_final[i] = input1.apPara["detune_final"][i];
        

        omegac[i] = omegarf[i]+detune[i]*2*pi;
        QL[i] = input1.rfPara["QL"][i];
        RoQ[i] = input1.rfPara["RoQ"][i];
        R[i] =QL[i]*RoQ[i];
        C[i] = QL[i]/R[i]/omegac[i];
        Leff[i] = dt*dt/(1-cos(dt*omegac[i]))/2/C[i];
        V_I_ref[i] = input1.rfPara["Vref_I"][i];
        V_Q_ref[i] = input1.rfPara["Vref_Q"][i];
        Ig_I_ref[i] = input1.rfPara["I_I_ref_ini"][i];
        Ig_Q_ref[i] = input1.rfPara["I_Q_ref_ini"][i];
        I_I_ref_ini[i] = input1.rfPara["I_I_ref_ini"][i];
        I_I_ref_final[i] = input1.rfPara["I_I_ref_final"][i];
        I_Q_ref_ini[i] = input1.rfPara["I_Q_ref_ini"][i];
        I_Q_ref_final[i] = input1.rfPara["I_Q_ref_final"][i];
        
        PA_cap[i] = sqrt(Ig_I_ref[i]*Ig_I_ref[i]+Ig_Q_ref[i]*Ig_Q_ref[i])*input1.apPara["PA_cap"][i];//
        nCav[i] = input1.apPara["nCav"][i];

        gii[i] = input1.apPara["gII"][i];
        gqq[i] = input1.apPara["gQQ"][i];
        gii_rec[i] = gii[i];
        gqq_rec[i] = gqq[i];
        I[i] = 0;
        Q[i] = 0;
#if 0
        std::cout<<Ig_I_ref[i]<<','<<std::endl;
        std::cout<<Ig_Q_ref[i]<<','<<std::endl;
        std::cout<<Leff[i]<<','<<std::endl;
        std::cout<<C[i]<<','<<std::endl;
        std::cout<<1/sqrt(Leff[i]*C[i])/2/pi<<','<<std::endl;
#endif
    }
    Urad0 = Urad0/(std::accumulate(nCav,nCav+nRF,0));
    // need to update the n_detune_ramp based on the required detune_mid of 56 MHz.
    n_detune_ramp = (n_detune_ramp_tot+detune_slow_factor*(detune_final[main_detune]-detune_mid[main_detune])/(detune_mid[main_detune]-detune_ini[main_detune])*n_detune_start)/(1+detune_slow_factor*(detune_final[main_detune]-detune_mid[main_detune])/(detune_mid[main_detune]-detune_ini[main_detune]));
    // now we need to recalculate the detune_mid for all the other HOMs
    for(int i = 0;i<nRF1+nRFc+nRF2;++i){
        detune_mid[i] = (detune_slow_factor*(n_detune_ramp-n_detune_start)*detune_final[i]+(n_detune_ramp_tot-n_detune_ramp)*detune_ini[i])/(detune_slow_factor*(n_detune_ramp-n_detune_start)+(n_detune_ramp_tot-n_detune_ramp));
    }
    double Trf = 2*pi/omegarf[0]; 
    double* t0s = new double[nBunch*nBeam]; // store the initial centers of all the bunches.
    double* gamma0s = new double[nBunch*nBeam];// store the initial center of gamma of all bunches.
    int nData = npRF*h[0];
    int nData_tot = nData+delay[0]+nSmp[0]; // number of data point, first 'delay+nSmp' number of points are for circle back the last 'delay+nSmp' points.
    int totN = nData_tot*(stride); // time, Ib[i-1], Ib[i], V0, Ig0, U0, V3, Ig3, U3, Vhom1, Uhom1, ...
    
    double Phi0 = -90;
    double eta = 1/GMTSQ-1/(Gamma0*Gamma0);
    int mainRF = input1.generalPara["mainRF"];
    double Qs = 0;
    double Vq = 0;
    // assume the first one or two RF are fundamental that provide main bucket
    for(int i = 0;i<nRF;++i){
        Vq += V_I_ref[i];
    }
    Qs = sqrt(h[mainRF]*atomicZ*fabs(Vq)*eta/(2*pi*Ek));

    double delta_hat = 0;
    switch(type_of_particle){
        case 0:
            delta_hat = sqrt(epsilon)*sqrt(omega0/(pi*Ek))*std::pow((h[mainRF]*atomicZ*fabs(Vq)/(2*pi*Ek*fabs(eta))),0.25);
            break;
        case 1:
            delta_hat = siglong/Gamma0;
            break;
        case 2:
            delta_hat = sqrt(epsilon)*sqrt(omega0/(pi*Ek))*std::pow((h[mainRF]*atomicZ*fabs(Vq)/(2*pi*Ek*fabs(eta))),0.25);    
            break;
        default:
            delta_hat = sqrt(epsilon)*sqrt(omega0/(pi*Ek))*std::pow((h[mainRF]*atomicZ*fabs(Vq)/(2*pi*Ek*fabs(eta))),0.25);
    }
    double t_hat = delta_hat/Qs*eta/omega0;
    double Ek_damp = Qs/input1.ringPara["Ek_damp"]; // the artificial damping, in unit of number of synchrotron oscillations
    double Ek_damp_always = Qs/input1.ringPara["Ek_damp_always"];

    std::cout<<"Qs"<<Qs<<std::endl;
    // initial coupled bunch modes
    int n_ini_CBI = input1.rfPara["n_ini_CBI"][0];
    std::vector<int> mode;
    std::vector<double> amp_gamma;
    std::vector<double> amp_time;
    mode.resize(n_ini_CBI);
    amp_gamma.resize(n_ini_CBI);
    amp_time.resize(n_ini_CBI);
    for(int i = 0;i<n_ini_CBI;++i){
        mode[i] = input1.rfPara["mu"][i];
        amp_gamma[i] = input1.rfPara["CBI_ini_amp"][i];
        amp_time[i] = amp_gamma[i]/Gamma0*fabs(eta)/Qs/(2*pi)*Trev;
        std::cout<<"mode "<<mode[i]<<" : "<<amp_gamma[i]<<" : "<<amp_time[i]<<std::endl;
    }

     // the comb filter related variables and vectors
    std::vector<std::vector<double>> Sp;
    std::vector<std::vector<double>> S;
    std::vector<double> epsilon_comb;
    std::vector<double> g_comb;
    std::vector<double> g_comb_rec;
    
    Sp.resize(nRF,std::vector<double>(nData_tot-nSmp[0],0));
    S.resize(nRF,std::vector<double>(nData_tot-nSmp[0],0));
    epsilon_comb.resize(nRF,0);
    g_comb.resize(nRF,0);
    g_comb_rec.resize(nRF,0);
    for(int i = 0;i<nRF;++i){
        epsilon_comb[i] = input1.apPara["epsilon_comb"][i];
        g_comb[i] = input1.apPara["g_comb"][i];
        g_comb_rec[i] = input1.apPara["g_comb"][i];
    }

    double coeff_decay_long = 2*Trev/t_rad_long;
    double coeff_excite_long = siglong*sqrt(3.0)*sqrt(4*Trev/t_rad_long);
    double* data1 = new double[totN]; // time, Ib1[i-1], Ib1[i],Ib2[i-1], Ib2[i], V0, Ig0, U0, V3, Ig3, U3, Vhom1, Uhom1, ..

    double* sintable = new double[nData_tot*(nRF1+nRFc+nRF2)];
    double* costable = new double[nData_tot*(nRF1+nRFc+nRF2)];
    double* tempI = new double[nSmp[0]*(nRF1+nRFc+nRF2)];
    double* tempQ = new double[nSmp[0]*(nRF1+nRFc+nRF2)];
    double* tempIb = new double[2*nBeam*nData_tot];

    double* rand_table = new double[nBeam*nBunch*nPar]; // store the random numbers for quantum excitation in electron update function.
    double* store = new double[nData*(stride)*int(N/step)]; // store the V,I info
    double* store_f_cav = new double[int(N/step)*(nRF1+nRFc+nRF2)]; // store the V, f, info
    double* store_M1_all = new double[int(N*nBunch*2*nBeam)];// store the centroids of each bunch, for all turns.
    double* store_M2 = new double[int(N/step)*nBunch*3*nBeam];
    double* store_M2_all = new double[nBunch*3*nBeam*N];// store all the second order moments.
    double* store_par = new double[int(N/step)*nBunch*2*nBeam*nPar];
    double* k = new double[5*(nRF1+nRFc+nRF2)]; //for voltage update

    // get the factors for the equation that updates the V(i) based on Ig,Ib,V(i-1),U(i-1)
    for (int i = 0;i<nRF1+nRFc+nRF2;++i){
        k[i*5] = 1-dt/R[i]/C[i];
        k[i*5+1] = dt/C[i];
        k[i*5+2] = -dt/C[i]/2;
        k[i*5+3] = k[i*5+2];
        k[i*5+4] = -dt/C[i]/Leff[i];
    }
#if 1
    std::cout<<"Ig_I_ref : "<<Ig_I_ref[0]<<std::endl;
    std::cout<<"Qs : "<<Qs<<std::endl;
    std::cout<<"theta hat : "<<t_hat*omegarf[0]/pi*180<<std::endl;
    std::cout<<"t hat : "<<t_hat<<std::endl;
    std::cout<<"delta hat : "<<delta_hat<<std::endl;
#endif
    // initialize the data
//#pragma omp parallel for 
    for(int i = 0;i<nData_tot;++i){
        //first data is time
        data1[i*stride] = dt*(i-delay[0]-nSmp[0])+dt/2;
        // second and third are Ib_(i-1) and Ib_i
        for(int j = 0;j<nBeam;++j){
            data1[i*stride+1+2*j] = 0;
            data1[i*stride+2+2*j] = 0;
        }
        // the rest of the 3*nRF are V, Ig, U of jth RF
        for(int j = 0;j<nRF;++j){
            data1[i*stride+j*3+1+2*nBeam] = 0;
            data1[i*stride+j*3+1+2*nBeam+1] = Ig_I_ref[j]*sin(data1[i*stride]*omegarf[j])+Ig_Q_ref[j]*cos(data1[i*stride]*omegarf[j]);
            data1[i*stride+j*3+1+2*nBeam+2] = 0;
            
            sintable[i*nRF+j] = sin(data1[i*stride]*omegarf[j]);        
            costable[i*nRF+j] = cos(data1[i*stride]*omegarf[j]);
        }
    }

    // put the last 'delay+nSmp' number of points at the beginning for further IQ calculation. 
    // assuming all RFs have same delay=delay[0]
    memcpy((void*)&data1[0],(void*)&data1[nData*stride],sizeof(double)*(delay[0]+nSmp[0])*stride);
    memcpy((void*)&sintable[0],(void*)&sintable[nData*nRF],sizeof(double)*(delay[0]+nSmp[0])*nRF);
    memcpy((void*)&costable[0],(void*)&costable[nData*nRF],sizeof(double)*(delay[0]+nSmp[0])*nRF);

    init_tempIQ(data1,sintable, costable, tempI, tempQ, I, Q, delay,nSmp,delay[0]+nSmp[0],nRF,stride);

    get_init_t0s(t0s,fill_step*Trf,Trf/2,Trev,nBeam,nBunch,beam_shift,h[0],mode,amp_time,n_ini_CBI);
    get_init_gamma0s(gamma0s,t0s,fill_step*Trf,Trf/2,Trev,Gamma0, nBeam,nBunch,beam_shift,h[0],mode,amp_gamma,n_ini_CBI);
    srand(0);
    std::cout<<"Initializing bunches."<<std::endl;
   
    for( int i = 0;i<nBunch*nBeam;++i){
        init_bunch(bunch,t0s[i],gamma0s[i],t_hat,delta_hat,nPar,nBunch,fill_count,nBeam);
        fill_count++;
    }
    for(int i = 0;i<nBunch*nBeam*nPar;++i){
        parFlag[i] = 1;
    }
    std::ofstream file;
    file.open("init.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&bunch[0],sizeof(double)*nPar*nBunch*nBeam*2);
    file.close();

    update_M1(bunch,M1,nPar,fill_count,nPar_survive,parFlag);
    update_M2(bunch,M1,M2,nPar,fill_count,nPar_survive,parFlag);
    auto t_start = omp_get_wtime(); 
    // start the tracking.
    for(int i = 0;i<N;++i){
        // using n_fill and n_q_ramp to ramp up the charge, artifically mimic the injecting process.
        for(int j = 0;j<nBeam;++j){
            q[j] = dqodn*(i-n_fill);
            q[j] = q[j]>0?q[j]:0;
            q[j] = q[j]<q0?q[j]:q0;
            q[j] = j==0?q[j]:-q[j];
        }
        
        // check if we need to turn on feedback.
        for(int rf = 0;rf<nRF;++rf){
            gii[rf] = i>fb_on[rf]?gii_rec[rf]:0;
            gqq[rf] = i>fb_on[rf]?gqq_rec[rf]:0;
            g_comb[rf] = i>fb_on[rf]?g_comb_rec[rf]:0;
        }
        // check to see if we need to turn off the Ek_damp
        if (i==n_Ek_damp_off){
            Ek_damp = Ek_damp_always;            
            std::cout<<"Turned off Ek_damp."<<std::endl;
        }
        // using qovermp_on as the flag for both proton and Au.
        qovermp_on = i>n_fill?1:0;
        qoverme_on = i>n_fill?1:0;
        // ramp the current
        if(i>=n_I_ramp_start & i<=n_I_ramp_end){
            current_rampping(i,n_I_ramp_start,n_I_ramp_end,nRF,dt,Ig_I_ref,Ig_Q_ref,I_I_ref_ini,I_I_ref_final,I_Q_ref_ini,I_Q_ref_final);
        }
        /// tune the frequencies of cavities, namely change the RLC circuit parameters
        if(i>=n_detune_start&i<=n_detune_ramp){
            detune_rampping(i,n_detune_start,n_detune_ramp,nRF,dt,omegac,omegarf,QL,C,R,Leff,k,detune,detune_ini,detune_mid);
        }
        if(i>=n_detune_ramp&i<=n_detune_ramp_tot){
            detune_rampping(i,n_detune_ramp,n_detune_ramp_tot,nRF,dt,omegac,omegarf,QL,C,R,Leff,k,detune,detune_mid,detune_final);
        }
        // find the time index of each particle
        update_index_neighbor(bunch, weight, dt, Trev, fill_count, nPar,h[0]*npRF,delay[0]+nSmp[0],nBunch, npRF, beam_shift, h[0],index,neighbor,parFlag);
        // update Ib
        updateIb(bunch,data1,weight,dt,q,fill_count,nPar,nData_tot,delay[0]+nSmp[0],stride,nBeam, index,neighbor,parFlag);
        // update the V and Ig
        updateVI(data1,k,nData_tot,dt,Ig_I_ref,Ig_Q_ref, V_I_ref, V_Q_ref,PA_cap,R,gii,gqq,I,Q,sintable, costable, tempI, tempQ, Sp,S,epsilon_comb,g_comb,delay,nSmp,nRF,nRF1,nRFc,nRF2,stride,nBeam);
        // put the last part of the VI data to the front of VI array, for IQ calculation. 
        memcpy((void*)&data1[0],&(data1[nData*stride]),sizeof(double)*(delay[0]+nSmp[0])*stride);
        
        if(delay[0]!=0){
            for(int iRF=0;iRF<nRF;++iRF){
                std::copy(S[iRF].end()-delay[0],S[iRF].end(),S[iRF].begin());
                std::copy(Sp[iRF].end()-delay[0],Sp[iRF].end(),Sp[iRF].begin());
            }
        }
        

        get_rand_table(rand_table,nBunch,nPar);
        //get_rand_table1(rand_table,nBunch,nPar,i);
        if(i>=n_dynamicOn){
            dynamicOn=1;
        }
        if(dynamicOn == 1 & type_of_particle==1){
            updateParticle_e(bunch,data1, weight, M1, rand_table,Gamma0,eta,dt,Trev,Ek_damp,Urad0,coeff_decay_long, coeff_excite_long,fill_count,nPar,nRF, nHOM,nBeam,nRF1,nRFc,nRF2,stride,nCav, index, neighbor);
        }

        if(dynamicOn == 1 & type_of_particle!=1){
            //std::cout<<i<<','<<q[0]<<std::endl;
            updateParticle(type_of_particle, bunch,data1, weight, M1, Gamma0,eta,dt,Trev,Ek_damp,Urad0,h[0],fill_step, fill_count,nPar,nRF, nHOM,nBeam,nRF1,nRFc,nRF2,stride,nCav,index, neighbor,parFlag);
        }

        update_M1(bunch,M1,nPar,fill_count,nPar_survive,parFlag);
        
        update_M2(bunch,M1,M2,nPar,fill_count,nPar_survive,parFlag);
        memcpy((void*)&store_M1_all[i*nBunch*nBeam*2],(void*)&M1[0],sizeof(double)*nBunch*nBeam*2);
        memcpy((void*)&store_M2_all[i*nBunch*nBeam*3],(void*)&M2[0],sizeof(double)*nBunch*nBeam*3);
        // store the data per 'step' turns.
        if(i%step == 0){
            memcpy((void*)&store[i/step*nData*stride],(void*)&data1[(delay[0]+nSmp[0])*stride],sizeof(double)*nData*stride);
            memcpy((void*)&store_f_cav[i/step*nRF],(void*)&omegac[0],sizeof(double)*nRF);
            update_M1(bunch,M1,nPar,fill_count,nPar_survive,parFlag);
            update_M2(bunch,M1,M2,nPar,fill_count,nPar_survive,parFlag);
            memcpy((void*)&store_M2[i/step*nBunch*nBeam*3],(void*)&M2[0],sizeof(double)*nBunch*nBeam*3);
            memcpy((void*)&store_par[i/step*nBunch*nBeam*2*nPar],(void*)&bunch[0],sizeof(double)*nBunch*nBeam*2*nPar);
        }
#if 1
        if(i%(N/10)==0){        
            std::cout<<"Completed "<< double(i)/N*100<<"%..."<<std::endl;
            std::cout<<"Charge per Bunch : "<<q[0]*nPar<<std::endl;
            for(int i = 0;i<nRF;++i){
                std::cout<<"detune : "<< detune[i]<<std::endl;
            }
        }
#endif
#if 0
        for(int bunchidx = 0;bunchidx<fill_count;++bunchidx){
            if (M2[bunchidx]>Trf*Trf){
                std::cout<<"Beam Lost! Bunch "<<bunchidx<<" lost at "<<i<<" turn!"<<std::endl;
                i = N; // finish the tracking
                break;
            } // if the rms bunch length is longer than the bucket, consider it lost
        }
#endif
    }
    update_M1(bunch,M1,nPar,fill_count,nPar_survive,parFlag);
    update_M2(bunch,M1,M2,nPar,fill_count,nPar_survive,parFlag);
    auto t_end = omp_get_wtime(); 
    std::cout<<"Total running time : "<<(t_end-t_start) <<" [s]. "<<std::endl;
    std::cout<<"Average time for one turn : "<< (t_end-t_start)*1000000/N<<" [us]. "<<std::endl;

    file.open("data.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store[0],sizeof(double)*nData*stride*int(N/step));
    file.close();

    file.open("f_cav.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store_f_cav[0],sizeof(double)*nRF*int(N/step));
    file.close();

    file.open("M2.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store_M2[0],sizeof(double)*nBunch*nBeam*3*int(N/step));
    file.close();

    file.open("par.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store_par[0],sizeof(double)*nPar*nBunch*nBeam*2*int(N/step));
    file.close();

    file.open("M1.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&M1[0],sizeof(double)*nBunch*nBeam*2);
    file.close();

    file.open("M1_all.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store_M1_all[0],sizeof(double)*N*nBunch*nBeam*2);
    file.close();

    file.open("M2_all.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&store_M2_all[0],sizeof(double)*N*nBunch*nBeam*3);
    file.close();

    file.open("first.bin", std::ios::binary);
    file.seekp(0);
    file.write((char*)&first_archive[0],sizeof(double)*(N-n_fill)*2);
    file.close();
    
    if(csv_out == 1){
        t_start = omp_get_wtime(); 
        // write the centroids data
        std::ofstream M1_all_csv;
        M1_all_csv.open("M1_all.txt");
        M1_all_csv<<"Turn , Bunch_idx , t0_i [second] , Gamma_i"<<std::endl;
        for(int i = 0;i<N;++i){
            for(int j = 0;j<nBunch*nBeam;++j){
                M1_all_csv<<i<<','<<j<<','<<store_M1_all[i*2*nBunch*nBeam+j*2]<<','<<store_M1_all[i*2*nBunch*nBeam+j*2+1]<<std::endl;
            }
        }
        M1_all_csv.close();

        //
        std::cout<<"Time spent writing csv file is : "<<omp_get_wtime()-t_start<<" [s]. "<<std::endl;
    }
    std::cout<<"Finish"<<std::endl;
    return 0;
}
