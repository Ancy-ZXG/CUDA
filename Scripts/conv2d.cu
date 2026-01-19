#define TILE 16
#define KERNEL 3
#define RADIUS (KERNEL / 2)

__constant__ float d_kernel[KERNEL * KERNEL];

extern "C" __global__
void conv2d(const float* input, float* output, int W, int H)
{
    // block / thread 索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    // Shared memory（带 halo）
    __shared__ float tile[TILE + 2*RADIUS][TILE + 2*RADIUS];

    // Shared memory 中的坐标
    int sx = tx + RADIUS;
    int sy = ty + RADIUS;

    // 1️⃣ 加载中心区域
    if (x < W && y < H)
        tile[sy][sx] = input[y * W + x];
    else
        tile[sy][sx] = 0.0f;

    // 2️⃣ 加载 halo（上下左右）
    if (tx < RADIUS) {
        // left
        tile[sy][sx - RADIUS] =
            (x >= RADIUS) ? input[y * W + x - RADIUS] : 0.0f;
        // right
        tile[sy][sx + TILE] =
            (x + TILE < W) ? input[y * W + x + TILE] : 0.0f;
    }

    if (ty < RADIUS) {
        // top
        tile[sy - RADIUS][sx] =
            (y >= RADIUS) ? input[(y - RADIUS) * W + x] : 0.0f;
        // bottom
        tile[sy + TILE][sx] =
            (y + TILE < H) ? input[(y + TILE) * W + x] : 0.0f;
    }

    __syncthreads();  // ★ 极其关键

    // 3️⃣ 卷积计算
    float sum = 0.0f;
    if (x < W && y < H) {
        for (int ky = -RADIUS; ky <= RADIUS; ky++) {
            for (int kx = -RADIUS; kx <= RADIUS; kx++) {
                float v = tile[sy + ky][sx + kx];
                float w = d_kernel[(ky + RADIUS)*KERNEL + (kx + RADIUS)];
                sum += v * w;
            }
        }
        output[y * W + x] = sum;
    }
}
