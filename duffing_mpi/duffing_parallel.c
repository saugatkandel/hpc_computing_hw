#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "mpi.h"
#include "funcs.h"


int main(int argc, char* argv[]){
    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set some default parameter values for convenience.
    // All the individual threads utilize these values.
    double alpha = 1.;
    double sigma = 0.5;
    int M = 100;
    int N = 1000;
    double T = 10;
    long int seed; 
    char *fname = "prob_parallel.out";
    
    // Input parameters
    if (argc > 1)
        sigma = (double)atof(argv[1]);
    if (argc > 2)
        M = (int)atol(argv[2]);
    if (argc > 3)
        N = (int)atol(argv[3]);
    if (argc > 5)
        T = (double)atof(argv[5]);
    
    if (rank == 0){
    
        // No need to regenerate the random seed for every thread.
        if (argc > 4)
            seed = (long int)atol(argv[4]);
        FILE* urand = fopen("/dev/urandom" , "r") ;
        fread(&seed , sizeof(long int), 1, urand);
        fclose(urand);
        
        // No need to create a new output file for every thread.
        if (argc > 6){
            fname = argv[6];
        }
#ifdef VERBOSE
        // Printing out the param values only once (and not once per thread)
        printf("Parameters are:\n");
        printf("apha: %f\n", alpha);
        printf("sigma: %f\n",sigma);
        printf("Number of time steps (M): %d\n", M);
        printf("Number of trials (N): %d\n", N);
        printf("Random seed is (+rank for mpi): %ld\n", seed);
        printf("Total terminal time (T): %f\n", T);
        printf("Output filename is %s\n", fname);
#endif
    }
    
    // actually initialize the random num generator for every thread
    MPI_Bcast (&seed , 1 , MPI_LONG_INT, 0 , MPI_COMM_WORLD);
    srand48(seed + rank);
    
    // making sure all the threads complete up to this step
    MPI_Barrier(MPI_COMM_WORLD);
    
    // assigning individual number of trials to each thread.
    // if N is a multiple of the num of threads, then localN = N / size
    // Otherwise, the lower ranked threads get the remainder jobs one by one
    int localN = ((N % size) > rank ? 1 : 0) + N / size;
    
#ifdef VERBOSE
    printf("size %d, rank %d, localN %d\n", size, rank, localN);
#endif
    double dt = (double) T / M;
    
    // setting up the array that counts the number of successful trials for each
    // t/10 timestep, for each thread.
    int out_len = M / 10 + 1;
    int* counts = (int*)malloc(sizeof(int) * out_len);

    // the array that accumulates the results. only need to initialize for rank 0.
    int* counts_all = NULL;
    if (rank == 0){
        counts_all = (int*)malloc(sizeof(int) * out_len * size);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // starting the actual timing.
    clock_t begin = clock();
    
    // the workhorse function that runs the trials
    single_thread_run(counts, alpha, sigma, dt, M, localN, out_len);
    
    // accumulates the results into counts_all
    MPI_Gather(counts, out_len, MPI_INT, counts_all, out_len, MPI_INT, 0, MPI_COMM_WORLD);
    
    // calculating the probabilities and writing them to an output file
    if (rank == 0){
        double* probs_final = (double*)malloc(sizeof(double) * out_len);
        
        for (int i=0; i < out_len; i++){
            probs_final[i] = 0;
            for (int j=0; j < size; j++){
                probs_final[i] += (double)counts_all[j * out_len + i] / N;
            }
        }
        
        FILE* output = fopen(fname, "w"); // Output file
        fwrite(probs_final, sizeof(double), out_len, output);
        fclose(output);
        
        printf("runtime %lf\n", (double)(clock()-begin)/CLOCKS_PER_SEC);
        free(probs_final);
    }
    
    free(counts);
    free(counts_all);
    
    MPI_Finalize();
    return 0;
}