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
    const int N = 512;
    const int M = 55;
    const double Xmin = -15.;
    const double Xmax = 15.;
    const double dt = 0.01;
    const double dx = (Xmax - Xmin)/(N-1);
    const double nuNyq = 1/(2*dx);
    //const double dp = 2.0*M_PI/(dx*N);
    const double dp = (N - 1)/(N*(Xmax - Xmin));
    std::vector<double> coordinate(N);
    fftw_complex *func_in = new fftw_complex[N];
    fftw_complex *func_out = new fftw_complex[N];
    fftw_plan plan_fwd, plan_bwd;
    plan_fwd = fftw_plan_dft_1d(N, func_in, func_out, FFTW_FORWARD, FFTW_MEASURE);
    plan_bwd = fftw_plan_dft_1d(N, func_out, func_in, FFTW_BACKWARD, FFTW_MEASURE);

    double X = 0.0;
    for (int i = 0; i < N; ++i)
    {
        X = Xmin+dx*i;
        coordinate[i] = X;
        func_in[i][0] = exp(-X*X);
        /*double X = 0.0;
        if(i < N/2)
        {
            X = dx*i;
            coordinate[N/2+i] = X;
            func_in[N/2+i][0] = exp(-X*X);
        }
        else
        {
            X = dx*i-dx*N;
            coordinate[i-N/2] = X;
            func_in[i-N/2][0] = exp(-X*X);
        }*/
        //double X = dx*i;
        //func_in[i][0] = exp(-X*X);
        func_in[i][1] = 0.;
    }

    std::cout<<func_in[1][0]<<' '<<func_in[N/2][0]<<' '<<func_in[N][0]<<std::endl;

    double *p = new double[N];
    /*for(int i = 0; i < N/2; ++i)
    {
        p[i] = dp*(double)i;
    }
    for(int i = N/2; i < N; i++)
    {
        p[i] = -dp*(N-i);
    }*/
    for(int i = 0; i < N; ++i)
    {
        p[i] = -nuNyq + dp*N;
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = exp(-V(coordinate[j])*dt)*func_in[j][0];
        }
        fftw_execute(plan_fwd);
        for (int j = 0; j < N; j++)
        {
            double re = func_out[j][0];
            double im = func_out[j][1];
            double phase = - double(j)*M_PI;
            func_out[j][0] = re*cos(phase) - im*sin(phase);
            func_out[j][1] = re*sin(phase) + im*cos(phase);
        }
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
            norma = norma + (func_in[j][0]*func_in[j][0])*dx;
        }


        double member = 1./sqrt(norma);

        for (int j = 0; j < N; ++j)
        {
            func_in[j][0] = func_in[j][0]*member;
        }
    }

    std::cout<<func_out[1][0]<<' '<<func_out[N/2][0]<<' '<<func_out[N][0]<<std::endl;
    std::cout<<func_in[1][0]<<' '<<func_in[N/2][0]<<' '<<func_in[N][0]<<std::endl;

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
