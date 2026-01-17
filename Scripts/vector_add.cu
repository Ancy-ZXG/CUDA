// vector_add.cu
extern "C" __global__
void vectorAdd(float* a, float* b, float* c, int N)
{
    extern __shared__ float buf[];
    float* sa = buf;
    float* sb = buf + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        sa[threadIdx.x] = a[idx];
        sb[threadIdx.x] = b[idx];
    }

    __syncthreads();

    if (idx < N) {
        c[idx] = sa[threadIdx.x] + sb[threadIdx.x];
    }
}
