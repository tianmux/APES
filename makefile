tmake: ./src/main.cpp 
	icpc ./src/main.cpp -o APES -qopenmp -qopt-report5 -qopt-report-phase=openmp,vec  -qopt-zmm-usage=high -std=c++14 -O3  -g -xCORE-AVX512  -I /opt/intel/advisor/include -liomp5 -lpthread -lm -ldl  -mkl=parallel 

