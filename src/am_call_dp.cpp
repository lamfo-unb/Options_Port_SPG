//Copyright (c) 2015 Intel Corporation
//All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of Intel Corporation
//   nor the names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior written
//   permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//   DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
//   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sys/time.h>
#include <time.h>
#include <tbb/scalable_allocator.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
#ifndef PI 
#define PI 3.141592653589793238462643
#endif

const double ACCURACY = 1.0e-6;
const int  ALIGNMENT = 1024;
//static int     OPT_N = 61*3*4*4*1024*32;
static long long     OPT_N = 16*17*512*1024*16ll;
const double RISKFREE = 0.02;
double sTime, eTime;

double second()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


inline double RandFloat(double low, double high, unsigned int *seed){
    double t = (double)rand_r(seed) / RAND_MAX;
    return (1.0 - t) * low + t * high;
}
__forceinline
double cnd_opt(double d){
 return 0.5 +0.5*erf(double(M_SQRT1_2)*d);
}

__forceinline
double n_opt(const double z) {
    return (1.0/sqrt(2.0*PI))*exp2(-0.5*z*z*M_LOG2E);
};

__forceinline
double european_call_opt( const double S, 
		const double X,
		const double r,
		const double q,
		const double sigma,
		const double time)
{
    double sigma_sqr = sigma*sigma;
    double time_sqrt = sqrt(time);
    double d1 = (log(S/X) + (r-q + 0.5*sigma_sqr)*time)/(sigma*time_sqrt);
    double d2 = d1-(sigma*time_sqrt);
    double call_price = S * exp2(-q*time*M_LOG2E)* cnd_opt(d1) - X * exp2(-r*time*M_LOG2E) * cnd_opt(d2);
    return call_price;
};
 __forceinline
double baw_scalaropt( const double S,
		 const double X, 
		 const double r,
		 const double b,
		 const double sigma,
		 const double time)
{
    double sigma_sqr = sigma*sigma;
    double time_sqrt = sqrt(time);
    double nn_1 = 2.0*b/sigma_sqr-1; 
    double m = 2.0*r/sigma_sqr;  
    double K = 1.0-exp2(-r*time*M_LOG2E); 
    double rq2 = 1/((-(nn_1)+sqrt((nn_1)*(nn_1) +(4.0*m/K)))*0.5);
    double rq2_inf = 1/(0.5 * ( -(nn_1) + sqrt(nn_1*nn_1+4.0*m)));
    double S_star_inf = X / (1.0 - rq2_inf);
    double h2 = -(b*time+2.0*sigma*time_sqrt)*(X/(S_star_inf-X));
    double S_seed = X + (S_star_inf-X)*(1.0-exp2(h2*M_LOG2E));
    double cndd1 = 0; 
    double Si=S_seed;         
    double g=1.;
    double gprime=1.0;
    double expbr=exp2((b-r)*time*M_LOG2E);
    for (  int no_iterations =0; no_iterations<100; no_iterations++) {
	double c  = european_call_opt(Si,X,r,b,sigma,time);
	double d1 = (log(Si/X)+(b+0.5*sigma_sqr)*time)/(sigma*time_sqrt);
    	double cndd1=cnd_opt(d1);
	g=(1.0-rq2)*Si-X-c+rq2*Si*expbr*cndd1;
//        if(fabs(g)  < ACCURACY) break;
	gprime=( 1.0-rq2)*(1.0-expbr*cndd1)+rq2*expbr*n_opt(d1)*(1.0/(sigma*time_sqrt));
	Si=Si-(g/gprime); 
    };
    double S_star = 0;
    if (fabs(g)>ACCURACY) { S_star = S_seed; }
    else { S_star = Si; };
    double C=0;
    double c  = european_call_opt(S,X,r,b,sigma,time);
    if (S>=S_star) {
	C=S-X;
    } 
    else {
	double d1 = (log(S_star/X)+(b+0.5*sigma_sqr)*time)/(sigma*time_sqrt);
	double A2 =  (1.0-expbr*cnd_opt(d1))* (S_star*rq2); 
	C=c+A2*pow((S/S_star),1/rq2);
    };
    return (C>c)?C:c;
};

void main()
{
    unsigned long long start_cyc;
    unsigned long long end_cyc;

    double S = 100;   double X = 100;     double sigma = 0.20;
    double r = 0.08;  double b = -0.04;   double time = 0.25;
    start_cyc = _rdtsc();
    double res = baw_scalaropt(S,X,r,b,sigma,time); 
    end_cyc = _rdtsc();
    cout << " Call price using Barone-Adesi Whaley approximation Optimized = " 
	 << std::setprecision(7) << res << endl << "  cycles consumed is " << end_cyc - start_cyc  << endl;
#ifdef _OPENMP
    	kmp_set_defaults("KMP_AFFINITY=scatter,granularity=fine");
	int ThreadNum = omp_get_max_threads();
	omp_set_num_threads(ThreadNum); 
#else
	int ThreadNum = 1; 
#endif
	long long OptPerThread = OPT_N / ThreadNum;
	long long mem_size = sizeof(double) * OptPerThread; 
	setlocale(LC_ALL,"");
	printf("Pricing American Options using BAW Approximation in double precision.\n");
	printf("Compiler Version  = %d\n", __INTEL_COMPILER/100);
	printf("Release Update    = %d\n", __INTEL_COMPILER_UPDATE);
	printf("Build Time        = %s %s\n", __DATE__, __TIME__);
	printf("Input Dataset     = %lld\n", OPT_N);
	printf("Worker Threads    = %d\n\n", ThreadNum);
	int threadID = 0; 
#pragma omp parallel 
{
#ifdef _OPENMP
	threadID = omp_get_thread_num();
#else
	threadID = 0; 
#endif

	double *CallResult = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *CallResult2 = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *StockPrice = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *OptionStrike = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *OptionYears = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *CostofCarry = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);
	double *Volatility = (double *)scalable_aligned_malloc(mem_size, ALIGNMENT);

        unsigned int seed = 123;
	unsigned int thread_seed = seed + threadID;
	for(long long i = OptPerThread-1; i > -1; i--)
	{
		CallResult[i] = 0.0;
		StockPrice[i]    = RandFloat(5.0, 30.0, &thread_seed);
		OptionStrike[i]  = RandFloat(1.0, 100.0, &thread_seed);
		OptionYears[i]   = RandFloat(0.25, 2.0, &thread_seed);
		CostofCarry[i]   = RandFloat(0.02, 0.05, &thread_seed);
		Volatility[i]   = RandFloat(0.10, 0.60, &thread_seed);
	}
#pragma omp barrier
#pragma omp master
	sTime = second();
#ifdef SCALAR
#else
#pragma vector nontemporal (CallResult)
#pragma simd
#pragma vector aligned
#endif
	for (int opt = 0; opt < OptPerThread; opt++)
	{

		double T = OptionYears[opt];
		double S = StockPrice[opt];
		double X = OptionStrike[opt];
		double b = CostofCarry[opt];
		double v = Volatility[opt];

		CallResult[opt] = baw_scalaropt(S,X,RISKFREE,b, v, T);
	}
#pragma omp barrier
#pragma omp master
	{
		eTime = second();
        	printf("Completed pricing %7.5f million options in %7.5f seconds:\n", OPT_N/1e6, eTime-sTime);
        	printf("%s version runs at %7.5f million options per second.\n",(ThreadNum > 1)?"Parallel":"serial", OPT_N/(1e6*(eTime-sTime)));
	}
	scalable_aligned_free(CallResult);
	scalable_aligned_free(CallResult2);
	scalable_aligned_free(StockPrice);
	scalable_aligned_free(OptionStrike);
	scalable_aligned_free(OptionYears);
	scalable_aligned_free(CostofCarry);
	scalable_aligned_free(Volatility);
}
};
