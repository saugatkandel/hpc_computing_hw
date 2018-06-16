#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <complex.h>

#include "funcs.h"

int main(int argc, char* argv[]){
    
    int i = 1;
    
    // Required command line inputs
    int N = (int)atol(argv[i++]); // Number of grid points per dimension
    double c1 = (double)atof(argv[i++]); // equation coefficient c1
    double c3 = (double)atof(argv[i++]); // equation coefficient c3
    int iterations = (int)atol(argv[i++]); // Total number of iterations (dt * T) - should be much larger than T

    // Fixed parameters
    int T = 10000; // Number of time units
    double dt = (double)T / iterations; // timestep per iteration
    double L = 128. * M_PI; // length of domain on each side
    
    
    char *fname = "CGL.out";
        
    if (argc > 5){
        T = (int)atol(argv[i++]);
    }
    if (argc == 7){
        fname = argv[i++];
    }
    printf("Parameters are:\n");
    printf("Number of grid points: %d\n", N);
    printf("Coefficient c1: %f\n",c1);
    printf("Coefficient c3: %f\n", c3);
    printf("Number of iterations (M): %d\n", iterations);
    printf("Total system timesteps (T) : %d\n", T);
    printf("Output file name: %s\n", fname);
    
    FILE* output = fopen(fname, "w"); // Output file

    long int seed; 
    FILE* urand = fopen("/dev/urandom" , "r") ;
    fread(&seed , sizeof(long int), 1, urand);
    fclose(urand);
    
    srand48(seed);
    
    clock_t begin = clock();
    
    // Initializing the array A with random complex numbers between [-1.5, 1.5] + i[-1.5, 1.5].
    complex *A;
    A = malloc(N * N * sizeof(complex));    
    for (int i=0; i < N; i++){
        for (int j=0; j < N; j++){
            A[i * N + j] = 3 * drand48() - 1.5 + I * (3 * drand48() - 1.5);
        }
    }
    
    // Initialize some reusable parameters
    // Also allocate space for the fft procedure.
    initialize(N, c1, c3, L);
    
    // Writing to the file at steps 0, 1000, 2000, ...
    fwrite(A, sizeof(complex), N*N, output);
    for (int i=0; i<10; i++){
        run_iterations(A, iterations / 10, dt);
        fwrite(A, sizeof(complex), N*N, output);
    }
    
    printf("\n\nruntime %g\n\n", (double)(clock()-begin)/CLOCKS_PER_SEC);
    fclose(output);
    
    // free up the memory assignments
    finalize();
    free(A);
    
    return 0;
    }
