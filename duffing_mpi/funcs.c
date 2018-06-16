#include "funcs.h"

void box_muller(double *r){
    double u1 = sqrt(-2 * log(drand48()));
    double u2 = 2 * M_PI * drand48();
    
    
    r[0] = u1 * cos(u2);
    r[1] = u1 * sin(u2);
}

void polar_marsaglia(double* r){
    double v1, v2, w, x;
    do{
        v1 = 2 * drand48() - 1;
        v2 = 2 * drand48() - 1;
        w = pow(v1, 2) + pow(v2, 2);
    }while( w> 1);
    
    x = sqrt(-2 * log(w) / w);
    r[0] = 0.1 * x * v1;
    r[1] = 0.1 * x * v2;
}

int test_probability(int x, int y, double alpha){
    double test1 = sqrt(pow(x - alpha, 2) + y * y);
    double test2 = sqrt(pow(x + alpha, 2) + y * y);

    int prob1 =  test1 <= alpha / 2 ? 1: 0;
    int prob2 =  test2 <= alpha / 2 ? 1: 0;
    return prob1 + prob2;
}

void initialize_rands(double* rand_nums, int size){
    // using polar_marsaglia
    double v1, v2, w1, w2;
    
    for (int i=0; i < size; i=i+2){
        do{
            v1 = 2 * drand48() - 1;
            v2 = 2 * drand48() - 1;
            w1 = pow(v1, 2) + pow(v2, 2);
        }while(w1>1);
        w2 = sqrt(-2 * log(w1) / w1);
        rand_nums[i] = 0.1 * w2 * v1;
        if (i + 1 < size){
            rand_nums[i+1] = 0.1 * w2 * v2;
        }
    }
}

void single_thread_run(int* counts, double alpha, double sigma, double dt, int M, int localN, int out_len){
    
    double x_old, y_old;
    double x_new, y_new;
    double y_term1, y_term2;
    
    double* rand_nums = (double*)malloc(sizeof(double) * (M + 1));
    
    for (int j=0; j< out_len; j++){
        counts[j] = 0;
    }
    
    for (int j=0; j <localN; j++){
        
        // create all the normally distributed random numbers required for this trial.
        initialize_rands(rand_nums, M+1);
        x_old = rand_nums[0];
        y_old = 0;
        
        counts[0] += test_probability(x_old, y_old, alpha);
        for(int k=0; k < M; k++){
            
            x_new = x_old + y_old * dt;
            y_term1 = ((pow(alpha,2) - pow(x_old, 2)) * x_old - y_old);
            y_term2 = sigma * x_old * sqrt(dt) * rand_nums[k + 1];
            y_new = y_old +  y_term1 + y_term2;
        
            if ((k + 1) % 10 == 0){
                counts[(k + 1) / 10] += test_probability(x_new, y_new, alpha);
            }   
            x_old = x_new;
            y_old = y_new;
        }    
    }
    
    free(rand_nums);
}

    
    


