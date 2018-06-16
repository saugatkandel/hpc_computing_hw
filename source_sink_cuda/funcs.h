#ifndef FUNCS_H
#define FUNCS_H
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265
#endif

// C version of the matlab/python linspace function. Populates the array gridpoints.
double linspace(double initval,
	      double finalval,
	      int N, 
	      double* gridpoints);


// Calculates the values of the function on the right hand side of the diffusion equation
// for a grid defined by xpoints and ypoints. Populats the array gvals. 

void gvalsCalc(double x_min, double x_max, double y_min, double y_max, 
                int xpoints, int ypoints, double lambda, double* gvals, 
                double* dx, double* dy);


// Initialize the "grid" 2d array to zero.
void init2dArray(int nx, int ny, double* grid);

// Print a 1-d array
void printArray(int N, double* array);

// Print a 2-d array in the MATLAB (column major) format.
void print2dArray(int nx, int ny, double* grid);

#endif
