
#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

struct timeval start, end;

//@@ Insert code to implement timer start
void timerStart() {
    gettimeofday(&start, NULL);
}

//@@ Insert code to implement timer stop
double timerStop() {
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    return elapsed;
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
    int inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);
  
    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType*)malloc(sizeof(DataType) * inputLength);
    hostInput2 = (DataType*)malloc(sizeof(DataType) * inputLength);
    hostOutput = (DataType*)malloc(sizeof(DataType) * inputLength);

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / RAND_MAX;
        hostInput2[i] = rand() / RAND_MAX;
    }  
  
    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**) &deviceInput1, sizeof(DataType) * inputLength);
    cudaMalloc((void**) &deviceInput2, sizeof(DataType) * inputLength);
    cudaMalloc((void**) &deviceOutput, sizeof(DataType) * inputLength);

    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    

    //@@ Initialize the 1D grid and block dimensions here
    dim3 dimGrid(ceil(inputLength / 32.0), 1);
    dim3 dimBlock(32, 1, 1);

    //@@ Launch the GPU Kernel here
    timerStart();
    vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    double time_elapsed = timerStop();
    printf("Kernel execution time: %.5lf\n", time_elapsed);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    bool correct = true;
    for (int i = 0; i < inputLength; i++) {
        if (hostOutput[i] != resultRef[i]) {
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
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    return 0;
}


