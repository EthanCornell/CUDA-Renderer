#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}





// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
// void exclusive_scan(int* input, int N, int* result) {
    
//     // CS149 TODO:
//     //
//     // Implement your exclusive scan implementation here.  Keep in
//     // mind that although the arguments to this function are device
//     // allocated arrays, this is a function that is running in a thread
//     // on the CPU.  Your implementation will need to make multiple calls
//     // to CUDA kernel functions (that you must write) to implement the
//     // scan.
//     // Allocate and initialize device memory

// }



//correct sol 1
// void exclusive_scan(int* input, int N, int* result) {
//     // Wrap raw pointers with device_ptr
//     thrust::device_ptr<int> dev_input(input);
//     thrust::device_ptr<int> dev_result(result);

//     // Perform the exclusive scan
//     thrust::exclusive_scan(dev_input, dev_input + N, dev_result);
// }

//Correct sol 2
__global__ void local_sum(int* device_result, int* partial_result, int range) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int l = index * range, r = l + range;
    for (int i=l+1; i<r; i++)
        device_result[i] += device_result[i-1];
    partial_result[index] = device_result[r-1];
}

__global__ void forward(int* device_result, int der) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * der * 2;
    device_result[index + (der*2) - 1] += device_result[index + der - 1];
}

__global__ void backward(int* device_result, int der) {
    int l = (blockIdx.x * blockDim.x + threadIdx.x) * der * 2 + der - 1;
    int r = l + der;
    int t = device_result[l];
    device_result[l] = device_result[r];
    device_result[r] += t;
}

__global__ void set_to_zero(int *device_result, int index) {
    device_result[index] = 0;
}

__global__ void to_result(int* device_result, int* device_start, int* partial_result, int range, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int l = index * range, r = l + range;
    if (r != n)
        device_result[r]  = device_start[r-1] + partial_result[index];
    for (int i=r-1; i>l; i--)
        device_result[i] = device_start[i-1] + partial_result[index];
} 

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    int n = nextPow2(length);
    const int totalBlocks = min(n, 8192);
    int* partial_result;
    cudaMalloc((void**)&partial_result, sizeof(int)*totalBlocks);
    int range = n/totalBlocks;
    local_sum<<<totalBlocks, 1>>>(device_start, partial_result, range);
    for (int der = 1; der < totalBlocks; der<<=1) {
        forward<<<totalBlocks/der/2, 1>>>(partial_result, der);
    }
    set_to_zero<<<1,1>>>(partial_result, totalBlocks-1);
    for (int der = totalBlocks/2; der >= 1; der >>= 1) {
        backward<<<totalBlocks/der/2, 1>>>(partial_result, der);
    }
    to_result<<<totalBlocks, 1>>>(device_result, device_start, partial_result, range, n);
    set_to_zero<<<1,1>>>(device_result, 0);
    cudaFree(partial_result);
}








//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}







// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
// int find_repeats(int* device_input, int length, int* device_output) {
    
//     // CS149 TODO:
//     //
//     // Implement this function. You will probably want to
//     // make use of one or more calls to exclusive_scan(), as well as
//     // additional CUDA kernel launches.
//     //    
//     // Note: As in the scan code, the calling code ensures that
//     // allocated arrays are a power of 2 in size, so you can use your
//     // exclusive_scan function with them. However, your implementation
//     // must ensure that the results of find_repeats are correct given
//     // the actual array length.

// }



__global__ void find_repeat_pos(int* a, int *eq, int range, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int l = index * range, r = min(l + range, length);
    int cnt = 0;
    if (index == 0) {
        l++;
    }
    for (int i=l; i<r; i++)
        cnt += (a[i] == a[i-1]);
    eq[index] = cnt;
}

__global__ void copy_to_output(int* out, int* a, int* pos, int range, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int l = index * range, r = min(l + range, length);
    int id = pos[index];
    if (index == 0)
        l++;
    for (int i=l; i<r; i++)
        if (a[i] == a[i-1]) {
            out[id++] = i-1;
        }
    pos[index] = id;
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    
    int n = nextPow2(length);
    const int totalBlocks = min(n, 8192);
    int *eq_pos, *eq_idx;
    cudaMalloc((void**)&eq_pos, sizeof(int)*totalBlocks);
    cudaMalloc((void**)&eq_idx, sizeof(int)*totalBlocks);
    int range = n/totalBlocks;
    find_repeat_pos<<<totalBlocks, 1>>>(device_input, eq_pos, range, length);
    int* out = new int[totalBlocks];
    exclusive_scan(eq_pos, totalBlocks, eq_idx);
    copy_to_output<<<totalBlocks, 1>>>(device_output, device_input, eq_idx, range, length);
    int ret;
    cudaMemcpy(&ret, &eq_idx[totalBlocks-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(eq_pos);
    cudaFree(eq_idx);
    return ret;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
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