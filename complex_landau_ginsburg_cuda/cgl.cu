#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cuComplex.h>
#include <complex.h>

// Not using shared memory. 
// Could be made much more efficient by using fft callbacks, 
// but that requires linking to static cuda libraries.
// See: https://devblogs.nvidia.com/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/


// Using a 16x16 grid of gpu threads.
const int num_threads = 16;

// these parameters stay constant throughout after setting. 
__device__ static cufftDoubleComplex dev_const_c1, dev_const_c3;
__device__ static double dev_const_L;
__device__ static int dev_N;

// Initialize the random kernels
__global__ void init_random(long int seed, curandState_t *states){
    int idx = threadIdx.x + blockIdx.x * num_threads;
    int idy = threadIdx.y + blockIdx.y * num_threads; 
    int id = idy * dev_N + idx;
    if (idx < dev_N && idy < dev_N){
        curand_init(seed, id, 0, &states[id]);
    }
}

// Generate the random values for the array A. 
// Also precalculate the x and y factors that we use in the derivative calculation.
__global__ void init_threads(curandState_t *states, cufftDoubleComplex* A,
                            double* xfactors, double* yfactors){
    int idx = threadIdx.x + blockIdx.x * num_threads;
    int idy = threadIdx.y + blockIdx.y * num_threads; 
    int id = idy * dev_N + idx;
    if (idx < dev_N && idy < dev_N){
        double real = 3 * curand_uniform(&states[id]) - 1.5;
        double imag = 3 * curand_uniform(&states[id]) - 1.5;
        A[id].x = real;
        A[id].y = imag;
        // precalculating the x and y factors.
        xfactors[id] = pow(fmin(idx, dev_N/1.0 - idx),2);
        yfactors[id] = pow(fmin(idy, dev_N/1.0 - idy),2);
    }
}

// To calculate the derivative (del2A), we need to 
// 1) perform fft
// 2) do some manipulation
// 3) perform ifft
// The fft and ifft are handled separately. 
// This function does only the in-between manipulation.
__global__ void del2A_between_fft(cufftDoubleComplex* out, double* xfactors, 
                                double* yfactors, cufftDoubleComplex* in){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
    int id = idy * dev_N + idx;
    if (idx < dev_N && idy < dev_N){
        double temp1 = -(xfactors[id] + yfactors[id]) / (dev_N * dev_N);
        out[id].x = temp1 * in[id].x;
        out[id].y = temp1 * in[id].y;
    }
}

// Final step in each rk4 iteration. 
// The paramter "a_iter" stays constant throughout the 4 rk4 steps, and is updated only at final step.
// "a_step" changes in between the steps
__global__ void rk4_step_finalize(cufftDoubleComplex *a_iter, cufftDoubleComplex* a_step,
                                    cufftDoubleComplex* a_out, double mult_const,
                                    cufftDoubleComplex* del2a){
    
    int idx = threadIdx.x + blockIdx.x * num_threads;
    int idy = threadIdx.y + blockIdx.y * num_threads; 
    int id = idy * dev_N + idx;
    if (idx < dev_N && idy < dev_N){
        cufftDoubleComplex term1, term2, term3, term4;
        double absval;

        term1 = a_step[id];
        absval = cuCabs(term1);
        term2 = cuCmul(dev_const_c1, del2a[id]);
        term2.x = term2.x * dev_const_L;
        term2.y = term2.y * dev_const_L;
        
        term3 = cuCmul(term1, dev_const_c3);
        term3.x = -term3.x * absval * absval;
        term3.y = -term3.y * absval * absval;

        term4 = cuCadd(term1, cuCadd(term2, term3));
        term4.x = term4.x * mult_const;
        term4.y = term4.y * mult_const;
        a_out[id] = cuCadd(a_iter[id], term4);
    }
}



int main(int argc, char* argv[]){
    
    // Required command line inputs
    int N = 128; // Number of grid points per dimension
    double c1 = 1.5; // equation coefficient c1
    double c3 = 0.25; // equation coefficient c3
    int iterations = 100000; // Total number of iterations (dt * T) - should be much larger than T

    // Fixed parameters
    int T = 10000; // Number of time units
    double dt = (double)T / iterations; // timestep per iteration
    double L = 128. * M_PI; // length of domain on each side
    
    
    const char* fname = "CGL.out";
    
    if (argc > 1){
        N = (int)atoi(argv[1]);
    }
    if (argc > 2){
        c1 = (double)atof(argv[2]);
    }
    if (argc > 3){
        c3 = (double)atof(argv[3]);
    }
    if (argc > 4){
        iterations = (int)atoi(argv[4]);
    }
    if (argc > 5){
        T = (int)atol(argv[5]);
    }
    if (argc == 6){
        fname = argv[6];
    }
#ifdef VERBOSE
    printf("Parameters are:\n");
    printf("Number of grid points: %d\n", N);
    printf("Coefficient c1: %f\n",c1);
    printf("Coefficient c3: %f\n", c3);
    printf("Number of iterations (M): %d\n", iterations);
    printf("Total system timesteps (T) : %d\n", T);
    printf("Output file name: %s\n", fname);
#endif
    
    FILE* output = fopen(fname, "w"); // Output file

    clock_t begin = clock();
    
    // Using an even grid 
    int num_blocks = N / num_threads + (N % num_threads ? 1 : 0);

    // Calculating some constant parameters for later use.
    cufftDoubleComplex const_c1;
    const_c1.x = 1;
    const_c1.y = c1;
    cufftDoubleComplex const_c3;
    const_c3.x = 1;
    const_c3.y = -c3;
    double const_L = (2. * M_PI / L) * (2 * M_PI / L);
    
    // Setting up the constants in the gpu memory.
    cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
    cudaMemcpyToSymbol(dev_const_c1, &const_c1, sizeof(cufftDoubleComplex));
    cudaMemcpyToSymbol(dev_const_c3, &const_c3, sizeof(cufftDoubleComplex));
    cudaMemcpyToSymbol(dev_const_L, &const_L, sizeof(double));
    
    // setting up the mesh
    dim3 meshBlocks(num_blocks, num_blocks);
    dim3 meshThreads(num_threads, num_threads);
    
    // initializing the random kernels in each thread
    curandState_t* dev_states;
    cudaMalloc((void**)&dev_states, N * N * sizeof(curandState));
    init_random<<<meshBlocks, meshThreads>>>(time(NULL), dev_states);
    
    // this is the actual grid array
    cufftDoubleComplex* A = (cufftDoubleComplex*)malloc(N * N * sizeof(cufftDoubleComplex));
    
    // more setup for gpu
    cufftDoubleComplex* dev_A;
    double* dev_xfactors;
    double* dev_yfactors;
    cudaMalloc((void**)&dev_A, N * N * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&dev_xfactors, N * N * sizeof(double));
    cudaMalloc((void**)&dev_yfactors, N * N * sizeof(double));

    // get the initial random values for A, and also calculate the x and y factors for later use in del2A calc
    init_threads<<<meshBlocks, meshThreads>>>(dev_states, dev_A, dev_xfactors, dev_yfactors);

    // setting up the fft
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
    
    // creating some temporary arrays for shifting data in between rk4 steps
    cufftDoubleComplex* dev_temp_a1;
    cufftDoubleComplex* dev_temp_a2;
    cudaMalloc((void**)&dev_temp_a1, N * N * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&dev_temp_a2, N * N * sizeof(cufftDoubleComplex));
    
    cufftDoubleComplex* dev_a_step;
    cudaMalloc((void**)&dev_a_step, N * N * sizeof(cufftDoubleComplex));
    
    
    cudaMemcpy(A, dev_A, N * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    fwrite(A, sizeof(cufftDoubleComplex), N*N, output);
    
    for (int i=0; i< 10; i++){
        for (int j=0; j < iterations / 10; ++j){

            // rk4 step 1
            cufftExecZ2Z(plan, dev_A, dev_temp_a1, CUFFT_FORWARD);
            del2A_between_fft<<<meshBlocks, meshThreads>>>(dev_temp_a2, dev_xfactors, dev_yfactors, dev_temp_a1);
            cufftExecZ2Z(plan, dev_temp_a2, dev_temp_a1, CUFFT_INVERSE);
            rk4_step_finalize<<<meshBlocks, meshThreads>>>(dev_A, dev_A, dev_a_step, dt / 4., dev_temp_a1);

            // step 2
            cufftExecZ2Z(plan, dev_a_step, dev_temp_a1, CUFFT_FORWARD);
            del2A_between_fft<<<meshBlocks, meshThreads>>>(dev_temp_a2, dev_xfactors, dev_yfactors, dev_temp_a1);
            cufftExecZ2Z(plan, dev_temp_a2, dev_temp_a1, CUFFT_INVERSE);
            rk4_step_finalize<<<meshBlocks, meshThreads>>>(dev_A, dev_a_step, dev_a_step, dt/3, dev_temp_a1);

            // step 3
            cufftExecZ2Z(plan, dev_a_step, dev_temp_a1, CUFFT_FORWARD);
            del2A_between_fft<<<meshBlocks, meshThreads>>>(dev_temp_a2, dev_xfactors, dev_yfactors, dev_temp_a1);
            cufftExecZ2Z(plan, dev_temp_a2, dev_temp_a1, CUFFT_INVERSE);
            rk4_step_finalize<<<meshBlocks, meshThreads>>>(dev_A, dev_a_step, dev_a_step, dt/2, dev_temp_a1);


            // step 4
            cufftExecZ2Z(plan, dev_a_step, dev_temp_a1, CUFFT_FORWARD);
            del2A_between_fft<<<meshBlocks, meshThreads>>>(dev_temp_a2, dev_xfactors, dev_yfactors, dev_temp_a1);
            cufftExecZ2Z(plan, dev_temp_a2, dev_temp_a1, CUFFT_INVERSE);
            rk4_step_finalize<<<meshBlocks, meshThreads>>>(dev_A, dev_a_step, dev_A, dt, dev_temp_a1);
        }
        cudaMemcpy(A, dev_A, N * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        fwrite(A, sizeof(cufftDoubleComplex), N*N, output);
    }
    printf("\n\nruntime %g\n\n", (double)(clock()-begin)/CLOCKS_PER_SEC);
    fclose(output);
    
    cufftDestroy(plan);
    cudaFree(dev_temp_a1);
    cudaFree(dev_temp_a2);
    cudaFree(dev_xfactors);
    cudaFree(dev_yfactors);
    cudaFree(dev_a_step);
    cudaFree(dev_A);

    free(A);
    return 0;
}