
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char **argv) {
    int inputLength;

    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    // @@ Insert code below to allocate Host memory for input and output
    // To launch kernel on different stream, Pinned-memory must be used. Why? 
    cudaMallocHost((void**)&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost((void**)&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost((void**)&hostOutput, inputLength * sizeof(DataType));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = (double) rand() / RAND_MAX;
        hostInput2[i] = (double) rand() / RAND_MAX;
    }  
  
    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**) &deviceInput1, sizeof(DataType) * inputLength);
    cudaMalloc((void**) &deviceInput2, sizeof(DataType) * inputLength);
    cudaMalloc((void**) &deviceOutput, sizeof(DataType) * inputLength);

    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    
    // Launch the vecAdd kernel on the GPU
    dim3 dimGrid(ceil(inputLength / 256));
    dim3 dimBlock(256);
    vecAdd<<<dimGrid, dimBlock, 0>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);

    // Wait until all issued CUDA calls are complete, i.e., hostOutput is correct
    cudaDeviceSynchronize();

    //@@ Insert code below to compare the output with the reference
    bool correct = true;
    resultRef = (DataType*)malloc(sizeof(DataType) * inputLength);
    for (int i = 0; i < inputLength; i++) {
        resultRef[i] = hostInput1[i] + hostInput2[i];
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-8) {
            correct = false;
            break;
        }
    }
    printf("Result is %s\n", correct ? "CORRECT" : "INCORRECT");


    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);
    return 0;
}


