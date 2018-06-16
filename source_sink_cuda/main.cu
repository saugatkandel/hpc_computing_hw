#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "funcs.h"
#include "handleerror.h"

// Using the RED-BLACK alternating update scheme. 
#define RED 0
#define BLACK 1

// Using a 2d grid of 16x16 threads. This grid should be safe in all modern gpus.
// Each thread corresponds to a point in the NxM grid.
const int num_threads = 16;

// declare constant memory on the device. these parameters stay constant throughout.
__device__ static int dev_N, dev_M;
__device__ static double dev_omega, dev_lambda, dev_dx, dev_dy;

// iterate is the gpu kernel that handles the gpu calls.
// the "red" parameter tells the function to alternate between red and black points in the grid.
__global__ void iterate(double* gvals, double *u, double* maxresid_per_block, int red);

// update is a gpu-only function that calculates the new "u" value
// and the residue per thread
__device__ void update(double* localu, double* localu_new, double* localg, 
                        double* resid_per_thread);

// initialize is a gpu-only function that copies the u and g values from the global memory
// to the local memory.
__device__ void initialize(double* localu, double* localu_new, double* localg, 
                            double* gvals, double* u);

// takes the residual obtained per thread and calculates the maximum value for each block
__device__ double getMaxResidPerBlock(double* resid_per_thread);
 
int main(int argc, char* argv[]){

    /* Setting up from the input parameters */
    //=====================================================================================
    int N = 128; // Number of y points
    double omega = 1.5; // Value of relaxation parameter.
    double tol = 1e-9;  // Tolerance. Default value is 1e-9.
    int maxiter = 1000; // Maximum number of iterations.
    const char* fname = "sources.out"; // Output file name
    double lambda = 100.; // lambda parameter.

    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        omega = atof(argv[2]);
    if (argc > 3){
        tol = atof(argv[3]);
    }
    if (argc > 4){
        maxiter = atoi(argv[4]);
    }
    if (argc > 5){
        fname = argv[5];
    }
    if (argc > 6){
        lambda = atof(argv[6]);
    }

    int M = 2 * N - 1;

#ifdef VERBOSE
    // Using a preprocessor flag to control verbose/debug output
    printf("Parameters are:\n");
    printf("number of y-points (N) : %d\n", N);
    printf("number of x-points (M) : %d\n", M);
    printf("omega : %f\n", omega);
    printf("tolerance: %f\n", tol);
    printf("max iterations: %d\n", maxiter);
    printf("Output fname: %s\n", fname);
    printf("lambda: %f\n", lambda);
#endif

double *gvals = (double*)malloc(M * N * sizeof(double));
double* u = (double*)malloc(N * M * sizeof(double));
double dx, dy, maxresid;

double x_min = -2., x_max = 2., y_min=-1, y_max=1;

gvalsCalc(x_min, x_max, y_min, y_max, M, N, lambda, gvals, &dx, &dy);
init2dArray(M, N, u);
maxresid = 1 + tol;

    // completed initial parameter setup.
    //========================================================================================
    clock_t begin = clock();

    // Setting the number of blocks. The blocks at the edges can have threads that are not utilized

    int num_blocks_x = M / num_threads + (M % num_threads ? 1 : 0);
    int num_blocks_y = N / num_threads + (N % num_threads ? 1 : 0);
#ifdef VERBOSE
    printf("threads %d\n", num_threads);
    printf("blocks %d %d\n", num_blocks_x, num_blocks_y);
#endif 

    
    double* resid_per_block_red = (double*)malloc(num_blocks_x * num_blocks_y * sizeof(double));
    double* resid_per_block_black = (double*)malloc(num_blocks_x * num_blocks_y * sizeof(double));

    // Copy the constants to the constant memory on the gpu
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_N, &N, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_M, &M, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_omega, &omega, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_lambda, &lambda, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_dx, &dx, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_dy, &dy, sizeof(double)));

    // initialize the global memory on the gpu.
    double* dev_u;
    double* dev_gvals;
    double* dev_resid_per_block; // the max residual is calculated up to per block within the gpu 

    HANDLE_ERROR(cudaMalloc((void**)&dev_u, N * M * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_gvals, N * M * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_resid_per_block, M * N * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_u, u, M * N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_gvals, gvals, M * N * sizeof(double), 
                            cudaMemcpyHostToDevice));
    
    // Setting up the blocks/threads grid.
    dim3 meshBlocks(num_blocks_x, num_blocks_y);
    dim3 meshThreads(num_threads, num_threads);
    
    int iter;
    for (iter=0; (iter < maxiter && maxresid > tol); iter++){
        
        // The actual update iterations. The RED and BLACK points are updated with sequential kernel calls.
        // Separately storing the residual per block from the red and black points. 
        iterate<<<meshBlocks, meshThreads>>>(dev_gvals, dev_u, dev_resid_per_block, RED);
        HANDLE_ERROR(cudaMemcpy(resid_per_block_red, dev_resid_per_block, num_blocks_x * num_blocks_y * sizeof(double), 
                                cudaMemcpyDeviceToHost));

        iterate<<<meshBlocks, meshThreads>>>(dev_gvals, dev_u, dev_resid_per_block, BLACK);
        HANDLE_ERROR(cudaMemcpy(resid_per_block_black, dev_resid_per_block, num_blocks_x * num_blocks_y * sizeof(double), 
                                cudaMemcpyDeviceToHost));
        
        // Comparing all residuals from the red and black points to calculate the maximum residual.
        maxresid = 0;
        for(int by=0; by<num_blocks_y; ++by){
            for(int bx=0; bx<num_blocks_x; ++bx){
                maxresid = fmax(resid_per_block_red[by * num_blocks_x + bx], maxresid);
                maxresid = fmax(resid_per_block_black[by * num_blocks_x + bx], maxresid);
            }
        }
#ifdef VERBOSE
        printf("iter %d maxresid %f\n", iter, maxresid);
#endif
    }
    // Copying back the final grid points
    HANDLE_ERROR(cudaMemcpy(u, dev_u, M * N * sizeof(double), 
                cudaMemcpyDeviceToHost));
   
    printf("runtime %g\n", (double)(clock()-begin)/CLOCKS_PER_SEC);

    FILE* output = fopen(fname, "w");
    fwrite(u, sizeof(double), N * M, output);
    fclose(output);

    free(u);
    free(gvals);
    free(resid_per_block_red);
    free(resid_per_block_black);
    HANDLE_ERROR(cudaFree(dev_u));
    HANDLE_ERROR(cudaFree(dev_gvals));
    HANDLE_ERROR(cudaFree(dev_resid_per_block));
    return 0;
}

// This is the workhorse function..
__global__ void iterate(double* gvals, double* u, double* resid_per_block, int red){
    
    // using the shared memory for information needed blockwise.
    __shared__ double localu[(num_threads + 2) * (num_threads + 2)];
    __shared__ double localg[num_threads * num_threads];

    // this is where the updated gridpoints are stored, before copying back to the global memory.
    __shared__ double localu_new[num_threads * num_threads];

    // initializing the array of residuals
    // The blocks at the edges can have unutilized threads, which don't update the corresponding
    // item in the array of residuals. 
    // To avoid using uninitialized values for the residuals, set them all to zero in advance.
    __shared__ double resid_per_thread[num_threads * num_threads];
    resid_per_thread[threadIdx.y * num_threads + threadIdx.x] = 0;

    int g_ix = blockIdx.x * num_threads + threadIdx.x;
    int g_iy = blockIdx.y * num_threads + threadIdx.y;
    int g_i = g_iy * dev_M + g_ix;
    // Ensuring that the calculation only happens at the threads that correspond to actual grid points.
    if (g_ix < dev_M && g_iy < dev_N){
        int t_i = threadIdx.y * num_threads + threadIdx.x; 
        
        // copying data from global to shared memory
        initialize(localu, localu_new, localg, gvals, u);
        __syncthreads();

        // This is an XOR trick to select only red or only black values per iteration.
        int cond2 = ((g_ix % 2 == 0) && (g_iy % 2 == 0));
        int cond3 = ((g_ix % 2 != 0) && (g_iy % 2 != 0));        
        if (red != (cond2 || cond3) ){
            // the actual updates
            update(localu, localu_new, localg, resid_per_thread);
        }

        // copy back to global memory.
        u[g_i] = localu_new[t_i];
        
        __syncthreads();

        // calculate the maximum residue per block from the individual threads.
        if (threadIdx.x == 0 && threadIdx.y == 0){
            resid_per_block[blockIdx.y * gridDim.y + blockIdx.x] = getMaxResidPerBlock(resid_per_thread);
        }
    }
    __syncthreads();
}

__device__ void update(double* localu, double* localu_new, double* localg, double* resid_per_thread){
    int l_ix = threadIdx.x + 1;
    int l_iy = threadIdx.y + 1;
    int l_dim = num_threads + 2;
    int l_i = l_iy * l_dim + l_ix;
    int t_i = threadIdx.y * num_threads + threadIdx.x;
    int g_ix = blockIdx.x * num_threads + threadIdx.x;
    int g_iy = blockIdx.y * num_threads + threadIdx.y;

    double usum = 0;

    // The conditionals are to make sure that we are only using values from within the grid (not periodic)
    // for the updates.
    if (g_iy > 0){ // for y =0, we update from points indexed as y-1
        usum += localu[(l_iy - 1) * l_dim + l_ix] - localu[l_i];
    }

    if (g_iy < (dev_N - 1)){ // for y = (N -1)
        usum += localu[(l_iy + 1) * l_dim + l_ix] - localu[l_i];
    }

    if (g_ix > 0){ // for x=0
        usum += localu[l_iy * l_dim + l_ix -1] - localu[l_i];
    }
    if (g_ix < (dev_M - 1)){ // for x = (M - 1)
        usum += localu[l_iy * l_dim + l_ix + 1] - localu[l_i];
    }
    
    double resid = 0.25 * (usum - dev_dx * dev_dy * localg[t_i]);
    localu_new[t_i] = localu[l_i] + dev_omega * resid;
    resid_per_thread[t_i] = resid;
}

__device__ void initialize(double* localu, double* localu_new, double* localg, double* gvals, double* u){
    int g_ix = blockIdx.x * num_threads + threadIdx.x;
    int g_iy = blockIdx.y * num_threads + threadIdx.y;
    int g_i = g_iy * dev_M + g_ix;
    
    // each block contains an additional two rows and two columns of ghost points at the edges.
    // the actual update points are indexed by l_ix.
    int l_ix = threadIdx.x + 1;
    int l_iy = threadIdx.y + 1;
    int l_dim = num_threads + 2;
    int l_i = l_iy * l_dim + l_ix;

    // For convenience, we assume that the grid is periodic for the initialization only.
    // the updates don't assume the same.
    int g_imx = (g_ix + dev_M -1) % dev_M;
    int g_imy = (g_iy + dev_N -1) % dev_N;
    int g_ipx = (g_ix + 1) % dev_M;
    int g_ipy = (g_iy + 1) % dev_N;
    int t_i = threadIdx.y * num_threads + threadIdx.x; 

    localu[l_i] = u[g_i];
    localu_new[t_i] = u[g_i];
    localg[t_i] = gvals[g_i];
    // The threads at the edges additionally initialize the adjacent ghost point.
    if (threadIdx.x == 0){
        localu[l_iy * l_dim] = u[g_iy * dev_M + g_imx];
    }
    if (threadIdx.x == num_threads - 1){
        localu[l_iy * l_dim + l_ix + 1] = u[g_iy * dev_M + g_ipx];
    }
    if (threadIdx.y == 0){
        localu[l_ix] = u[g_imy * dev_M + g_ix];
    }
    if (threadIdx.y == num_threads - 1){
        localu[(l_iy + 1) * l_dim + l_ix] = u[g_ipy * dev_M + g_ix];
    }
}

__device__ double getMaxResidPerBlock(double* resid_per_thread){
    double maxresid = 0;
        for (int ry=0; ry < num_threads; ry++){
            for (int rx=0; rx < num_threads; rx++){
                maxresid = fmax(maxresid, fabs(resid_per_thread[ry * num_threads + rx]));
            }
        }
    return maxresid;
}