#ifndef FUNCS_H
#define FUNCS_H
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323
#endif

/* Test implementations of box-muller and polar-marsaglia algorithms */
void box_muller(double* r);
void polar_marsaglia(double* r);

// Check whether the given xy point satisfies the probability condition specified
//      in the textbook
int test_probability(int x, int y, double alpha);

// Generate an array (of length size) of normally distributed random numbers
//      using the polar-marsaglia method.
void initialize_rands(double* rand_nums, int size);

/* 
The workhorse function that runs localN different simulations, of M steps each,
for each individual cpu thread. At each timestep t % 10 ==0, the function
checks whether the current xy point satisfies the probability check. If yes,
the accumulated count for that timestep increases by 1.

Parameters:

counts: the array that stores the accumulated count for each timestep t % 10 == 0

alpha, sigma: input parameters

dt: size of individual timestep.

M: number of timesteps.

localN: number of independent runs for this thread.

out_len: size of the counts array.
*/
void single_thread_run(int* counts, double alpha, double sigma, double dt, int M, int localN, int out_len);






#endif