#include "funcs.h"

void initialize(int N, double c1, double c3, double L){
    
    f_N = N;
    f_const_c1 = 1. + I * c1;
    f_const_c3 = 1. - I * c3;
    f_const_L = (2. * M_PI / L) * (2. * M_PI / L);
    
    f_a = fftw_malloc(N * N * sizeof(complex));
    f_temp_in = fftw_malloc(N * N * sizeof(complex));
    f_temp_out = fftw_malloc(N * N * sizeof(complex));
    f_del2a = fftw_malloc(N * N * sizeof(complex));
    
    f_p = fftw_plan_dft_2d(N, N, f_a, f_temp_out, FFTW_FORWARD, FFTW_ESTIMATE);
    f_pinv = fftw_plan_dft_2d(N, N, f_temp_in, f_del2a, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    f_xfactors = malloc(N * N * sizeof(double));
    f_yfactors = malloc(N * N * sizeof(double));
    
    int indx;
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            indx = i * N + j;
            f_yfactors[indx] = (double)(fmin(i, N - i) * fmin(i, N - i));
            f_xfactors[indx] = (double)(fmin(j, N - j) * fmin(j, N - j));
        }
    }
}

void finalize(){
    fftw_destroy_plan(f_p);
    fftw_destroy_plan(f_pinv);
    
    fftw_free(f_a);
    fftw_free(f_temp_in);
    fftw_free(f_temp_out);
    fftw_free(f_del2a);
    
    free(f_xfactors);
    free(f_yfactors);
}

void copy_array(complex* a_in, complex* a_out){
    int indx;
    for (int i=0; i<f_N; i++){
        for (int j=0; j<f_N; j++){
            indx = i * f_N + j;
            a_out[indx] = a_in[indx];
        }
    }
}

void run_iterations(complex* A, int num_iterations, double dt){
    complex *temp_a1, *temp_a2;
    temp_a1 = malloc(f_N * f_N * sizeof(complex));
    temp_a2 = malloc(f_N * f_N * sizeof(complex));
    
    // the RK4 steps
    for(int i=0; i<num_iterations; i++){
        runge_kutta_step(A, A, temp_a1, dt / 4);
        runge_kutta_step(A, temp_a1, temp_a2, dt / 3);
        runge_kutta_step(A, temp_a2, temp_a1, dt / 2);
        runge_kutta_step(A, temp_a1, A, dt);
    }
    free(temp_a1);
    free(temp_a2);
}

void runge_kutta_step(complex* a_iter, complex* a_step, 
                      complex* a_out, double mult_const){
    copy_array(a_step, f_a);
    calc_del2A();
    
    int indx;
    complex term1;
    complex term2;
    complex term3;
    double absval;
    for (int i=0; i<f_N; i++){
        for(int j=0; j<f_N; j++){
            indx = i * f_N + j;
            
            term1 = a_step[indx];
            absval = cabs(term1);
            term2 = f_const_c1 * f_del2a[indx] * f_const_L;
            term3 = f_const_c3 * absval * absval * term1;
            a_out[indx] = a_iter[indx] + mult_const * (term1 + term2 - term3);
        }
    }
    
}

void calc_del2A(){

    fftw_execute(f_p);
    
    int indx;
    for (int i=0; i<f_N; i++){
        for (int j=0; j<f_N; j++){
            indx = i * f_N + j;
            f_temp_in[indx] = - (f_xfactors[indx] + f_yfactors[indx]) * f_temp_out[indx] / (f_N * f_N);
        }
    }
    fftw_execute(f_pinv);
    
}

void print_2d_array(complex* grid){
    int indx;
  for (int i=0; i < f_N; i++){
      for (int j=0; j< f_N; j++){
          indx = i * f_N + j;
          printf("%f + %fi\t", creal(grid[indx]), cimag(grid[indx]));
      }
      printf("\n");
    }
}