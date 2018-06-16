#ifndef FUNCS_H
#define FUNCS_H

#define _XOPEN_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>
#include "fftw3.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Creating some global variables to store any data that is repeatedly used
// "f_" prefix indicates that the variable is declared as global in funcs.h

// array size per dimension
int f_N;  

// Since the array size remains constant throughout, we can reuse
// fftw plans. This saves a lot of time.
// To reuse fftw plans, we need to have created the fft input and output
// arrays beforehand.
complex *f_a, *f_temp_in, *f_temp_out, *f_del2a;
fftw_plan f_p, f_pinv;

// multipliers for the second derivative calculation
// They are assigned values in a way that avoids using fftshift
double *f_xfactors, *f_yfactors;

// constants related to the input paramters
complex f_const_c1, f_const_c3;
double f_const_L;

// Initialize all the global variables; allocate the required memory
void initialize(int N, double c1, double c3, double L);

// Utility function to copy an array into another
void copy_array(complex* a_in, complex* a_out);

// calculate the second derivative using the fft operations
void calc_del2A();

// the workhorse function that runs the set number of iterations
void run_iterations(complex* A, int num_iterations, double dt);

// individual runge-kutta update step
void runge_kutta_step(complex* a_iter, complex* a_step, 
                      complex* a_out, double mult_const);

// free memory allocated for global arrays
void finalize();

// utility function to print a complex 2d array
void print_2d_array(complex* grid);




#endif