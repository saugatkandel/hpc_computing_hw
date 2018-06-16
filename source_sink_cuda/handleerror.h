#ifndef HANDLE_ERROR_H
#define HANDLE_ERROR_H

static void HandleError( cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))

#endif