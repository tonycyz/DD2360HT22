#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

    //@@ Insert code below to compute histogram of input using shared memory and atomics
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > num_elements) return;

    // bin is a thread-local variable
    int bin = input[i]; // corresponding index of the integer input of the current thread

    // Create shared memory for each thread block.
    // the array shared_bins has a copy in each thread block 
    __shared__ unsigned int shared_bins[NUM_BINS]; // create an array to store the bins in shared memory
    
    if(threadIdx.x == 0) {
        for (int idx = 0; idx < NUM_BINS; idx++)
            shared_bins[idx] = 0;
    }

    __syncthreads();
    atomicAdd(&shared_bins[bin], 1);
    __syncthreads();

    // the #0 thread in each block is responsible for writing result to global mem
    if(threadIdx.x == 0) {
        for(int i = 0; i < NUM_BINS; i++)
            atomicAdd(&bins[i], shared_bins[i]); // shared_bins is local for each thread block
    } // bins is global in device memory

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < num_bins) {
        if(bins[i] > 127)
            bins[i] = 127;
    } 
}


int main(int argc, char **argv) {
  
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);
    
    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*)malloc(sizeof(unsigned int) * inputLength);
    hostBins = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BINS);
   
    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for(int i = 0; i < inputLength; i++) {
        hostInput[i] = rand() % NUM_BINS;
    }

    //@@ Insert code below to create reference result in CPU
    resultRef = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BINS);
    for(int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
    }
    for(int i = 0; i < inputLength; i++) {
        resultRef[hostInput[i]]++;
    }
    for(int i = 0; i < NUM_BINS; i++) {
        if(resultRef[i] > 127)
            resultRef[i] = 127;
    }


    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**)&deviceInput, sizeof(unsigned int) * inputLength);
    cudaMalloc((void**)&deviceBins, sizeof(unsigned int) * NUM_BINS);

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);
    
    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, sizeof(unsigned int) * NUM_BINS);
    
    //@@ Initialize the grid and block dimensions here
    dim3 threadsPerBlock(256);
    dim3 numBlocks(ceil((float)inputLength / threadsPerBlock.x));

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    //@@ Initialize the second grid and block dimensions here
    dim3 threadsPerBlock_1(256);
    dim3 numBlocks_1(ceil((float)inputLength / threadsPerBlock_1.x));

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<numBlocks_1, threadsPerBlock_1>>>(deviceBins, NUM_BINS);
    
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    bool correct = true;
    int err = 0;
    for(int i = 0; i < NUM_BINS; i++) {
        err = hostBins[i] - resultRef[i];
        if( err ) {
            correct = false;
            break;
        }
    }
    if(correct) printf("The results are CORRECT.\n"); else printf("The results are INCORRECT, error is %d.\n", err);

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}

