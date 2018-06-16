#ifndef FUNCS_H
#define FUNCS_H

#include <stdio.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"

// Function names self-explanatory. Not all functions are used.
double linspace(double initval, double finalval, int N, double* gridpoints);
void print_2d_array(int nx, int ny, double* grid);
void copy_2d_array(double* input, double* output, int nx, int ny);
void matmul(double* m1, double* m2, double* prod, int m1x, int m1y, int m2x, int m2y);


// initialize a square 2d array with all zeros
void init_2d_zeros(double* x, int N);

// set diagonal elements of a 2d array to 1
void set_diag_ones(double* m, int N);

// Utility function that first sets all zeros, then the diags to 1.
void set_zeros_ones(double* m, int N);


double* transpose(double* m1, int nx, int ny);

// For a column matrix of size 4NxN, ie consisting of four NxN matrices
// stacked vertically, this function transposes each of the NxN matrices
// individually, and returns the new 4NxN stack. 
void transpose_blockwise(double* input, double* output, int block_size);



void create_mat(double* m, int N);

// Utility function that initializes the matrices V, MRx, MRy, MLx, 
// and MLy. 
// Refer to Pg. 353 (Section 35.3) for the definitions of the individual 
// matrices. 
// Adopts the notation from the professor's matlab script.
void initMatrices(double* V_, double* MRx_, 
                  double* MRy_, double* MLx_, double* MLy_, 
                  int N, double x0, double x1, double dt, double gamma);
    








#endif