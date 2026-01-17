// matrix_mul.cu
#define TILE 16


//dim3 block(TILE, TILE);   // 16 × 16 = 256 threads
//dim3 grid(
//    (N + TILE - 1) / TILE,  // x 方向 block 数
//    (M + TILE - 1) / TILE   // y 方向 block 数
//);

//matMul<<<grid, block>>>(A, B, C, M, N, K);




extern "C" __global__
void matMul(const float* A, const float* B, float* C,
            int M, int N, int K)
{
    // block 和 thread 的二维索引
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // shared memory：每个 block 一块
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // 分块遍历 K 维
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // 加载 A / B 到 shared memory
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // ★ 非常关键

        // 当前 tile 内累加
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();  // ★ 防止下一轮覆盖
    }

    // 写回结果
    if (row < M && col < N)
        C[row * N + col] = sum;
}
