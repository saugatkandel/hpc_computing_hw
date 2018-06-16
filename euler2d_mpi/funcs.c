#include "funcs.h"

double linspace(double initval,
	      double finalval,
	      int N,
	      double* gridpoints)
{
  double diff = (finalval - initval) / N;

  gridpoints[0] = initval;
  for (int i=1; i<N; i++)
    gridpoints[i] = gridpoints[i-1] + diff;

  return diff;
}

void print_2d_array(int nx, int ny, double* grid)
{
  for (int i=0; i<nx; i++)
    {
      for (int j=0; j<ny; j++)
	printf("%e  ",grid[ny * i + j]);
      printf("\n");
    }
}

void copy_2d_array(double* input, double* output, int nx, int ny){
    
        for (int i=0; i<nx; i++){
            for(int j=0; j<ny; j++){
                output[i * ny + j] = input[i * ny + j];
                output[i * ny + j] = input[i * ny + j];
            }
        }
}

void init_2d_zeros(double* x, int N){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            x[i * N + j] = 0;
        }
    }
}


void create_mat(double* m, int N){
    m = (double*)malloc(sizeof(double) * N * N);
}

void matmul(double* m1, double* m2, double* prod, int m1x, int m1y, int m2x, int m2y){
    for(int i=0; i<m1x; i++){
        for(int j=0; j<m2y; j++){
            prod[i * m2y + j] = 0;
            for(int k=0; k<m1y; k++){
                prod[i * m2y + j] += m1[i * m1y + k] * m2[k * m2y + j];
            }
        }
    }
    
}

void transpose_blockwise(double* input, double* output, int block_size){
    int N = block_size;
    for(int i=0; i<block_size; i++){
        for(int j=0; j<block_size; j++){
            // Transposing all the four blocks simultaneously
            output[i * N + j] = input[j * N + i];
            output[(i + N) * N + j] = input[N * N  + j * N + i];
            output[(i + 2 * N) * N + j] = input[2 * N * N + j * N + i];
            output[(i + 3 * N) * N + j] = input[3 * N * N + j * N + i];
        }
    }
}

double* transpose(double* m1, int nx, int ny){
    double* m2 = (double*)malloc(sizeof(double) * nx * ny);
    for(int i=0; i<nx; i++){
        for(int j=0; j<ny; j++){
            m2[j * nx + i] = m1[i * nx + j];
        }
    }
    return m2;
}

void set_diag_ones(double *m, int N){
    for (int i=0; i<N; i++)
        m[i * N + i] = 1;
}

void set_zeros_ones(double *m, int N){
    init_2d_zeros(m, N);
    set_diag_ones(m, N);
}

void initMatrices(double* V_, double* MRx_, 
                  double* MRy_, double* MLx_, double* MLy_, 
                  int N, double x0, double x1, double dt, double gamma){
    
    double* x = (double*)malloc(sizeof(double) * N);
    double dx = linspace(x0, x1, N, x);
    
    // Using multidimensional matrix notation just for convenience. 
    // The pointers point to the same location as the input 
    // flattened matrix. 
    double (*V)[N] = (double (*)[4 * N])V_;
    double (*MRx)[4 * N] = (double (*)[4 * N])MRx_;
    double (*MRy)[4 * N] = (double (*)[4 * N])MRy_;
    double (*MLx)[4 * N] = (double (*)[4 * N])MLx_;
    double (*MLy)[4 * N] = (double (*)[4 * N])MLy_;
    
    // Creating the temporay D matrix makes the assignments 
    // for MRx, MRy, MLx, MLy easier.
    double (*D)[N] = malloc(sizeof(*D) * N);
    set_zeros_ones(*MLx, 4 * N);
    set_zeros_ones(*MLy, 4 * N);
    set_zeros_ones(*MRx, 4 * N);
    set_zeros_ones(*MRy, 4 * N);
    
    // Initializing all the matrices simultaneously
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            
            // Initializing the D matrix to zeros, 
            // then assigning other values as necessary.  
            D[i][j] = 0;
            if (j==i+1)
                D[i][j] = dt / 4 / dx;
            
            if(j==i-1)
                D[i][j] = -dt / 4 / dx;
            
            if(j==0 && i==N-1)
                D[i][j] = dt / 4 / dx;
            if(i==0 && j==N-1)
                D[i][j] = -dt / 4 / dx;

            // Note that the matrix V is a vertical stacking of the individual NxN matrices
            // {rho, u, v, p}
            V[i][j] = 2 / gamma * exp(-100 * (x[i]*x[i] + x[j]*x[j])); // rho
            V[N + i][j] = 0; // u
            V[2 * N + i][j] = 0; // v
            V[3 * N + i][j] = 2 * exp(-100 * (x[i]*x[i] + x[j]*x[j])); // p
            
            MRx[i][N+j] = -D[i][j];
            MRx[i + N][3 * N + j] = -D[i][j];
            MRx[i + 3 * N][N + j] = - gamma * D[i][j];
            
            MLy[i][2 * N + j] = D[i][j];
            MLy[i + 2 * N][3 * N + j] = D[i][j];
            MLy[i + 3 * N][2 * N + j] = gamma * D[i][j];
            
            MRy[i][2 * N + j] = -D[i][j];
            MRy[i + 2 * N][3 * N + j] = -D[i][j];
            MRy[i + 3 * N][2 * N + j] = -gamma * D[i][j];
            
            MLx[i][N + j] = D[i][j];
            MLx[i + N][3 * N + j] = D[i][j];
            MLx[i + 3 * N][N + j] = gamma * D[i][j];
   
        }
    }
    free(x);
    free(D);
}
