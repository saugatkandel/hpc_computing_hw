#include "funcs.h"

double linspace(double initval,
	      double finalval,
	      int N,
	      double* gridpoints)
{
  double diff = (finalval - initval) / (N-1);

  gridpoints[0] = initval;
  for (int i=1; i<N; i++)
    gridpoints[i] = gridpoints[i-1] + diff;

  return diff;
}

void gvalsCalc(double* xpoints,
	       double* ypoints,
	       int nx,
	       int ny,
	       double lambda,
	       double* gvals){
  double mfactor = 10. * lambda / sqrt(M_PI);
  double l2 = lambda * lambda;
  int indx;
  double xhigh, xhighsq, xlow, xlowsq, term1, term2, ysq;
  
  for (int i=0; i<nx; i++){
    for (int j=0; j<ny; j++){
      indx = i * ny + j;
      xlow = xpoints[i] - 1;
      xlowsq = xlow * xlow;
      
      xhigh = xpoints[i] + 1;
      xhighsq = xhigh * xhigh;

      ysq = ypoints[j] * ypoints[j];
      
      term1 = exp(-l2 * (xlowsq + ysq));
      term2 = exp(-l2 * (xhighsq + ysq));

      gvals[indx] = mfactor * term1 - mfactor * term2;
    }
  }
}

void init2dArray(int nx, int ny, double* grid)
{
  for (int i=0; i<nx; i++)
    {
      for (int j=0; j<ny; j++){
	grid[i * ny + j] = 0;
      }
    }
}	      

void printArray(int N, double* array)
{
  for (int i=0; i<N; i++)
    printf("%d %f\n", i, array[i]);
}


void print2dArray(int nx, int ny, double* grid)
{
  for (int i=0; i<nx; i++)
    {
      for (int j=0; j<ny; j++)
	printf("%f\t",grid[ny * i + j]);
      printf("\n");
    }
}


