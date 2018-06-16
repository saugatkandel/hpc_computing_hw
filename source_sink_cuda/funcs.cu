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
/*
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
  
  for (int i=0; i<ny; i++){
    for (int j=0; j<nx; j++){
      indx = i * nx + j;
      xlow = xpoints[j] - 1;
      xlowsq = xlow * xlow;
      
      xhigh = xpoints[j] + 1;
      xhighsq = xhigh * xhigh;

      ysq = ypoints[i] * ypoints[i];
      
      term1 = exp(-l2 * (xlowsq + ysq));
      term2 = exp(-l2 * (xhighsq + ysq));

      gvals[indx] = mfactor * term1 - mfactor * term2;
    }
  }
}
*/

void gvalsCalc(double x_min, double x_max, double y_min, double y_max, 
                int xpoints, int ypoints, double lambda, double* gvals, 
                double* dx, double* dy){

    *dx = (x_max - x_min) / (xpoints - 1);
    *dy = (y_max - y_min) / (ypoints - 1);
    double mfactor = 10. * lambda / sqrt(M_PI);
    double xval, xlow, xlowsq, xhigh, xhighsq, yval, ysq, term1, term2;
    
    for (int i=0; i < ypoints; i++){
      for (int j=0; j<xpoints; j++){

        xval = x_min + j * (*dx);
        xlow = xval - 1;
        xlowsq = xlow * xlow;
        xhigh = xval + 1;
        xhighsq = xhigh * xhigh;

        yval = y_min + i *  (*dy);
        ysq = yval * yval;

        term1 = exp(-lambda * lambda * (xlowsq + ysq));
        term2 = exp(-lambda * lambda * (xhighsq + ysq));

        gvals[i * xpoints + j] = mfactor * term1 - mfactor * term2;
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
  for (int i=0; i<ny; i++)
    {
      for (int j=0; j<nx; j++)
	printf("%f  ",grid[nx *i + j]);
      printf("\n");
    }
}


