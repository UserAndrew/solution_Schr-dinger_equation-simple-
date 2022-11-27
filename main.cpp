#include <iostream>
#include <fstream>
#include <fftw3.h>
#include <cmath>
#include <vector>

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
    const int M = 10000;
    const double dt = 0.01;
    const double dx = 0.1;
    const double Xmin = -409.55; // -4*r_osc
    const double Xmax = 409.55;  // 4*r_osc
    const double N1 = (Xmax - Xmin)/dx;
    const int N = pow(2,(int(log(N1)/log(2))+1));
    const double dp = (2*M_PI)/(dx*N);

    std::vector<double> coordinate(N);
    fftw_complex *func_in = new fftw_complex[N];
    fftw_complex *func_out = new fftw_complex[N];
    fftw_plan plan_fwd, plan_bwd;
    plan_fwd = fftw_plan_dft_1d(N, func_in, func_out, FFTW_FORWARD, FFTW_MEASURE);
    plan_bwd = fftw_plan_dft_1d(N, func_out, func_in, FFTW_BACKWARD, FFTW_MEASURE);

// теперь диапазон [Xmin;Xmax] другой, массив coordinate[] не содержит нуль
    double X = dx/2;
    for(int i = N/2; i < N; ++i)
    {
        coordinate[i] = X;
        func_in[i][0] = exp(-X*X);
        func_in[i][1] = 0.;
        X += dx;
    }

    X = -dx/2;
    for(int i = N/2-1; i >= 0; --i)
    {
        coordinate[i] = X;
        func_in[i][0] = exp(-X*X);
        func_in[i][1] = 0.;
        X -= dx;
    }

    double *p = new double[N];
    for(int i = 0; i < N/2; ++i)
    {
        p[i] = dp*i;
    }
    for(int i = N/2; i < N; ++i)
    {
        p[i] = -dp*(N-i);
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = exp(-V(coordinate[j])*dt)*func_in[j][0];
        }

        fftw_execute(plan_fwd);

        for (int j = 0; j < N; ++j)
        {
            func_out[j][0] = exp(-p[j]*p[j]*dt/2.)*func_out[j][0];
            func_out[j][1] = exp(-p[j]*p[j]*dt/2.)*func_out[j][1];
        }
        fftw_execute(plan_bwd);

        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = (1./N)*func_in[j][0];
        }

        double re_norma = 0;
        for (int j = 0; j < N; ++j)
        {
            re_norma = re_norma + (func_in[j][0]*func_in[j][0])*dx;
        }


        double re_member = 1./sqrt(re_norma);

        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = func_in[j][0]*re_member;
        }

        for (int j = 0; j < N; ++j)
        {
            func_in[j][1] = 0;
        }
    }

    std::ofstream _out("Analitical_solution.dat");
    _out.precision(4);
    for (int i = 0; i < N; ++i)
    {
        _out<<std::fixed<<Psi_solution(coordinate[i])<<'\t'<</*dx*i*/coordinate[i]<<std::endl;
    }

    std::ofstream _out_("Numerical_solution.dat");
    _out_.precision(4);
    for (int i = 0; i < N; ++i)
    {
        _out_<<std::fixed<<func_in[i][0]<<'\t'<<coordinate[i]<<std::endl;
    }

    delete [] p;
    delete [] func_out;
    delete [] func_in;
    return 0;
}
