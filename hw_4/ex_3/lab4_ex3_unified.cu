
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define DataType double

// define floating-point error tolerance
#define FP_ERR_TOL 1e-8

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows, int numAColumns, int numBRows, int numBColumns){
    int numCColumns = numBColumns;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // i = rowC * numCColumns + colC, colC < numCRows
    int rowC = i / numCColumns;
    int colC = i - rowC * numCColumns;
    
    C[i] = 0;    
    for(int idx = 0; idx < numAColumns; idx++) {
        C[i] += A[rowC * numAColumns + idx] * B[idx * numBColumns + colC];
    }
}

int main(int argc, char **argv) {
  
    DataType *uni_A; // The A matrix
    DataType *uni_B; // The B matrix
    DataType *uni_C; // The output C matrix
    DataType *resultRef; // The reference result
    
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = numAColumns;
    numBColumns = atoi(argv[3]);    
    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocManaged((void**)&uni_A, sizeof(DataType) * numARows * numAColumns);
    cudaMallocManaged((void**)&uni_B, sizeof(DataType) * numBRows * numBColumns);
    cudaMallocManaged((void**)&uni_C, sizeof(DataType) * numCRows * numCColumns);
    
    resultRef = (DataType*)malloc(sizeof(DataType) * numCRows * numCColumns);
  
    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for(int i = 0; i < numARows * numAColumns; i++) {
        uni_A[i] = (double) rand() / RAND_MAX;
    }
    for(int i = 0; i < numBRows * numBColumns; i++) {
        uni_B[i] = (double) rand() / RAND_MAX;
    }
    for(int r = 0; r < numCRows; r++) {
        for(int c = 0; c < numCColumns; c++) {
            resultRef[r*numCColumns + c] = 0.0;
            for(int i = 0; i < numAColumns; i++)
                resultRef[r*numCColumns+c] += uni_A[r*numAColumns+i] * uni_B[i*numBColumns+c];
        }
    }

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numCRows * numCColumns / 32.0), 1);
    dim3 dimBlock(32, 1, 1);

    //@@ Launch the GPU Kernel here
    gemm<<<dimGrid, dimBlock>>>(uni_A, uni_B, uni_C, numARows, numAColumns, numBRows, numBColumns);

    //@@ Insert code below to compare the output with the reference
    bool correct = true;
    for(int i = 0; i < numCRows * numCColumns; i++) {
        if(fabs(resultRef[i] - uni_C[i]) > FP_ERR_TOL) {
            correct = false;
            break;
        }
    }
    printf("GEMM result is %s\n", correct ? "CORRECT" : "INCORRECT");

    //@@ Free the GPU memory here
    cudaFree(uni_A);
    cudaFree(uni_B);
    cudaFree(uni_C);

    free(resultRef); 

    return 0;
}
