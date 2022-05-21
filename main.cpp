#include <iostream>
#include <fstream>
//using namespace std;
#include <fftw3.h>
#include <cmath>

double V(double x)
{
    return (-1.)/(sqrt(pow(x,2)+2));
}

double Psi_solution(double x)
{
    return exp(-sqrt(x*x+2))*(1+sqrt(x*x+2));
}

int main()
{
    const int N = 128;
    const double dt = 0.01;
    const double dx = 0.1;
    const double dp = 2.0*M_PI/(dx*N);
    fftw_complex *func_in = new fftw_complex[N];
    fftw_complex *func_out = new fftw_complex[N];
    fftw_plan plan_fwd, plan_bwd;
    plan_fwd = fftw_plan_dft_1d(N, func_in, func_out, FFTW_FORWARD, FFTW_MEASURE);
    plan_bwd = fftw_plan_dft_1d(N, func_out, func_in, FFTW_BACKWARD, FFTW_MEASURE);

    for (int i = 0; i < N; ++i)
    {
        double X = dx*i;
        func_in[i][0] = exp(-X*X);
        func_in[i][1] = 0.;
    }

    double *p = new double[N];
    for(int i = 0; i < N; ++i)
    {
        p[i] = dp*(double)i;
    }


    delete [] p;
    delete [] func_out;
    delete [] func_in;
    return 0;
}
