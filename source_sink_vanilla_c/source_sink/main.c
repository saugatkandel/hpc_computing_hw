#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <getopt.h>
#include <time.h>

#include "funcs.h"

// This is the workhorse function that calculates the residuals using the SOR algorithm.
int getIterations(int ny,
		  int nx,
		  double omega,
		  double tol,
		  double lambda,
		  int maxiter,
		  double* runtime){

  double *xpoints = malloc(sizeof(xpoints) * nx);
  double *ypoints = malloc(sizeof(ypoints) * ny);
  double *gvals = malloc(nx * ny * sizeof(gvals));
  double *u = malloc(nx * ny * sizeof(u));
  double dx, dy, resid, maxresid;
  int iter;
  clock_t start_t, end_t;

  dx = linspace(-2., 2., nx, xpoints);
  dy = linspace(-1., 1., ny, ypoints);
  gvalsCalc(xpoints, ypoints, nx, ny, lambda, gvals);
  
  init2dArray(nx, ny, u);
  maxresid = 1 + tol;
  resid = 0;
  start_t = clock();
  for(iter=0; (maxresid > tol && iter < maxiter); iter++){

    maxresid = 0;
    for (int i=0; i < nx; i++){
      for (int j=1; j< ny-1; j++){
	double usum = 0;
	
	usum += u[i * ny + j - 1] - u[i * ny + j];
	usum += u[i * ny + j + 1] - u[i * ny + j];
	
	if (i != 0)
	  usum += u[(i-1) * ny + j] - u[i * ny + j];
	
	if (i != (nx - 1))
	  usum += u[(i + 1) * ny + j] - u[i * ny +j];
	
	resid = 0.25 * usum - dx * dy * gvals[i * ny + j] / 4.;
	maxresid = fmax(fabs(resid), maxresid);
	u[i * ny + j] += omega * resid;
      }
    }
  }

  end_t = clock();
  *runtime = (double) (end_t - start_t) / CLOCKS_PER_SEC;
  
  free(gvals);
  free(u);
  free(xpoints);
  free(ypoints);
  
  return iter;
}

int main(int argc, char* argv[]){
  
  int N = -1; // Number of y points - required command line input 
  int M; // Number of x points. Set to be 2N -1.
  double omega = -1.; // Value of relaxation parameter. Required parameter (between 0-2).
  double tol = 1e-9; // Tolerance. Default value is 1e-9. Can be adjusted with command line.
  double lambda = 100.; // lambda parameter. Can be adjusted with command line.
  int maxiter = 1000; // Maximum number of iterations. Can be adjusted with command line.
  bool print_iter_time = false; // Whether to print the time required for each iteration.

  int iterations;
  double runtime;


  extern char *optarg;
  extern int optind;

  int p = -1;

  int i;
  
  while((i = getopt(argc, argv, "N:o:t:l:m:p")) != -1)
    switch(i){
    case 'N':
      N = (int)atol(optarg);
      break;
    case 'o':
      omega = (double)atof(optarg);
      break;
    case 't':
      tol = (double)atof(optarg);
      break;
    case 'l':
      lambda = (double)atof(optarg);
      break;
    case 'm':
      maxiter = (int)atol(optarg);
      break;
    case 'p':
      print_iter_time = true;
      break;
    default:
      printf("USAGE default\n");
      break;
    }

   // Check mandatory parameters:
   if (N < 1) {
      printf("-N is mandatory! Should be a positive integer\n");
      exit(1);
   }
   M = 2 * N - 1;

   
   if (omega < 0 || omega > 2){
     printf("-o is mandatory. Omega should be between 0 and 2\n");
     exit(1);
   }
   

   
   iterations = getIterations(N, M,  omega, tol, lambda, maxiter, &runtime);

   printf("Iterations %d\n", iterations);
   if (print_iter_time)
     printf("runtime %g\n", runtime);

   return 0;
}
