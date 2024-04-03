#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

// return GB/sec
float GBPerSec(int bytes, float sec)
{
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
saxpy_kernel(int N, float alpha, float *x, float *y, float *result)
{

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N)
        result[index] = alpha * x[index] + y[index];
}

// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
void saxpyCuda(int N, float alpha, float *xarray, float *yarray, float *resultarray)
{

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!
    float *device_x = nullptr;
    float *device_y = nullptr;
    float *device_result = nullptr;

    //
    // CS149 TODO: allocate device memory buffers on the GPU using cudaMalloc.
    //
    // We highly recommend taking a look at NVIDIA's
    // tutorial, which clearly walks you through the few lines of code
    // you need to write for this part of the assignment:
    //
    // https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
    //

    // Allocate device memory
    cudaMalloc((void **)&device_x, N * sizeof(float));
    cudaMalloc((void **)&device_y, N * sizeof(float));
    cudaMalloc((void **)&device_result, N * sizeof(float));

    // Copy input arrays to the GPU
    cudaMemcpy(device_x, xarray, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, N * sizeof(float), cudaMemcpyHostToDevice);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO: copy input arrays to the GPU using cudaMemcpy
    //

    // Kernel invocation with timing
    double kernelStartTime = CycleTimer::currentSeconds();

    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    //
    // CS149 TODO: copy result from GPU back to CPU using cudaMemcpy
    //

    cudaDeviceSynchronize();
    double kernelEndTime = CycleTimer::currentSeconds();
    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Kernel execution time: %.3f ms\n", 1000.f * kernelDuration);

    // Copy result from GPU back to CPU
    cudaMemcpy(resultarray, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();


    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess)
    {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
                errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));

    //
    // CS149 TODO: free memory buffers on the GPU using cudaFree
    //
    // Free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void printCudaInfo()
{

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

// Question 1. What performance do you observe compared to the sequential
// CPU-based implementation of SAXPY (recall your results from saxpy on Program
// 5 from Assignment 1)?

// One would need to run both the CPU-based implementation and their new CUDA-based
// implementation to observe and compare their performance. Typically, the CUDA implementation
// is found to be significantly faster for large arrays due to parallel processing on the GPU.

// Question 2. Compare and explain the difference between the results provided
// by two sets of timers (timing only the kernel execution vs. timing the entire
// process of moving data to the GPU and back in addition to the kernel
// execution). Are the bandwidth values observed roughly consistent with the
// reported bandwidths available to the different components of the machine?
// (You should use the web to track down the memory bandwidth of an NVIDIA T4
// GPU. Hint:
// https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf.
// The expected bandwidth of memory bus of AWS is 4 GB/s, which does not match
// that of a 16-lane PCIe 3.0. Several factors prevent peak bandwidth, including
// CPU motherboard chipset performance and whether or not the host CPU memory
// used as the source of the transfer is “pinned” — the latter allows the GPU to
// directly access memory without going through virtual memory address
// translation. If you are interested, you can find more info here:
// https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory)

// The difference in results between timing only the kernel execution vs. timing the entire process (including data
// transfer to and from the GPU) is crucial. Data transfer over PCIe can be a bottleneck due to its lower bandwidth
// compared to GPU memory bandwidth. The observed bandwidth values for kernel execution should be higher and more
// consistent with the GPU's memory bandwidth. In contrast, the overall process including data transfer might align
// more closely with PCIe bandwidth limitations, affecting total performance.