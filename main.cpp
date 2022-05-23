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
    const int M = 100;
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
    std::cout<<func_in[1][0]<<' '<<func_in[N/2][0]<<' '<<func_in[N][0]<<std::endl;

    double *p = new double[N];
    for(int i = 0; i < N; ++i)
    {
        p[i] = dp*(double)i;
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = exp(-V(j*dx)*dt)*func_in[j][0];
        }
        fftw_execute(plan_fwd);

        for (int j = 0; j < N; ++j)
        {
            func_out[j][0] = exp(-p[j]*p[j]*dt/2.)*func_out[j][0];
        }
        fftw_execute(plan_bwd);

        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = (1./N)*func_in[j][0];
        }

        double norma = 0;
        for (int j = 0; j < N; ++j)
        {
            norma += (dx*(func_in[j][0]));
        }


        double member = norma/sqrt(norma);

        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] *= member;
        }
    }

    std::cout<<func_out[1][0]<<' '<<func_out[N/2][0]<<' '<<func_out[N][0]<<std::endl;
    std::cout<<func_in[1][0]<<' '<<func_in[N/2][0]<<' '<<func_in[N][0]<<std::endl;

    std::ofstream _out("Analitical_solution.dat");
    _out.precision(4);
    for (int i = 0; i < N; ++i)
    {
        _out<<std::fixed<<Psi_solution(i*dx)<<'\t'<<dx*i<<std::endl;
    }

    std::ofstream _out_("Numerical_solution.dat");
    _out_.precision(4);
    for (int i = 0; i < N; ++i)
    {
        _out_<<std::fixed<<func_in[i][0]<<'\t'<<dx*i<<std::endl;
    }

    delete [] p;
    delete [] func_out;
    delete [] func_in;
    return 0;
}
