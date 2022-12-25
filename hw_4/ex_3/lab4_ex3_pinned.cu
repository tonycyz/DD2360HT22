
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
  
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
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
    cudaMallocHost((void**)&hostA, sizeof(DataType) * numARows * numAColumns);
    cudaMallocHost((void**)&hostB, sizeof(DataType) * numBRows * numBColumns);
    cudaMallocHost((void**)&hostC, sizeof(DataType) * numCRows * numCColumns);
    resultRef = (DataType*)malloc(sizeof(DataType) * numCRows * numCColumns);
  
    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for(int i = 0; i < numARows * numAColumns; i++) {
        hostA[i] = (double) rand() / RAND_MAX;
    }
    for(int i = 0; i < numBRows * numBColumns; i++) {
        hostB[i] = (double) rand() / RAND_MAX;
    }
    for(int r = 0; r < numCRows; r++) {
        for(int c = 0; c < numCColumns; c++) {
            resultRef[r*numCColumns + c] = 0.0;
            for(int i = 0; i < numAColumns; i++)
                resultRef[r*numCColumns+c] += hostA[r*numAColumns+i] * hostB[i*numBColumns+c];
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**) &deviceA, sizeof(DataType) * numARows * numAColumns);
    cudaMalloc((void**) &deviceB, sizeof(DataType) * numBRows * numBColumns);
    cudaMalloc((void**) &deviceC, sizeof(DataType) * numCRows * numCColumns);

    // @@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(DataType) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBRows * numBColumns, cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numCRows * numCColumns / 32.0), 1);
    dim3 dimBlock(32, 1, 1);

    //@@ Launch the GPU Kernel here
    gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(DataType) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    bool correct = true;
    for(int i = 0; i < numCRows * numCColumns; i++) {
        if(fabs(resultRef[i] - hostC[i]) > FP_ERR_TOL) {
            correct = false;
            break;
        }
    }
    printf("GEMM result is %s\n", correct ? "CORRECT" : "INCORRECT");

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //@@ Free the CPU memory here
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    free(resultRef); 

    return 0;
}
