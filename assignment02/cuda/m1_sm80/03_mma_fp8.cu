// 03_mma_fp8.cu —— 完整版：单条 m16n8k32 e4m3 MMA
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// ============================================================
// Fragment 装载：A 行主序，B 列主序
// ============================================================
__device__ void load_A_manual(const uint8_t* sA, uint32_t a[4]) {
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = gid + 8 * (i & 1);
        int col = tig * 4 + 16 * (i >> 1);
        a[i] = *reinterpret_cast<const uint32_t*>(sA + row * 32 + col);
    }
}

__device__ void load_B_manual(const uint8_t* sB, uint32_t b[2]) {
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int n = gid;
        int k = tig * 4 + 16 * i;
        b[i] = *reinterpret_cast<const uint32_t*>(sB + n * 32 + k);
    }
}

__device__ void store_D_manual(float* gD, float d[4]) {
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = gid + 8 * (i >= 2);
        int col = tig * 2 + (i & 1);
        gD[row * 8 + col] = d[i];
    }
}

// ============================================================
// Kernel：一条 m16n8k32 e4m3 MMA，使用 shared memory
// ============================================================
__global__ void mma_fp8_kernel(const __nv_fp8_e4m3* __restrict__ gA,
                               const __nv_fp8_e4m3* __restrict__ gB,
                               float* __restrict__ gD) {
    __shared__ __nv_fp8_e4m3 sA[16 * 32];
    __shared__ __nv_fp8_e4m3 sB[8 * 32];   // 列主序：N=8, K=32

    for (int i = 0; i < 16 * 32; i++) sA[i] = gA[i];
    for (int i = 0; i < 8 * 32;  i++) sB[i] = gB[i];
    __syncthreads();

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t a[4];
    uint32_t b[2];

    load_A_manual(reinterpret_cast<const uint8_t*>(sA), a);
    load_B_manual(reinterpret_cast<const uint8_t*>(sB), b);

    // ---------- 单条 MMA ----------
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );

    store_D_manual(gD, d);
}

// ============================================================
// CPU 参考（行主序）
// ============================================================
static void cpu_mma(const float* A, const float* B, float* D) {
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            float sum = 0.f;
            for (int k = 0; k < 32; k++)
                sum += A[m * 32 + k] * B[k * 8 + n];
            D[m * 8 + n] = sum;
        }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <seed>\n", argv[0]);
        return 1;
    }
    unsigned seed = (unsigned)atoi(argv[1]);
    srand(seed);

    // ---------- 主机端数据 ----------
    float hA[16 * 32];
    float hB_row[32 * 8];   // 行主序，用于 CPU 参考
    float hB_col[32 * 8];   // 列主序，用于拷贝到设备
    float hD[16 * 8];
    float hRef[16 * 8];

    // A 行主序
    for (int i = 0; i < 16 * 32; i++)
        hA[i] = (float)(rand() % 16);

    // B：同时生成行主序和列主序
    for (int k = 0; k < 32; k++) {
        for (int n = 0; n < 8; n++) {
            float val = (float)(rand() % 16);
            hB_row[k * 8 + n] = val;   // 行主序
            hB_col[n * 32 + k] = val;  // 列主序
        }
    }

    cpu_mma(hA, hB_row, hRef);

    // ---------- 转 fp8 ----------
    __nv_fp8_e4m3 hA_fp8[16 * 32];
    __nv_fp8_e4m3 hB_fp8[32 * 8];
    for (int i = 0; i < 16 * 32; i++) hA_fp8[i] = __nv_fp8_e4m3(hA[i]);
    for (int i = 0; i < 32 * 8;  i++) hB_fp8[i] = __nv_fp8_e4m3(hB_col[i]);  // 列主序

    // ---------- 设备内存 ----------
    __nv_fp8_e4m3 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA_fp8)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB_fp8)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(hD)));

    CUDA_CHECK(cudaMemcpy(dA, hA_fp8, sizeof(hA_fp8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_fp8, sizeof(hB_fp8), cudaMemcpyHostToDevice));

    // 1 block, 32 线程（1 warp）
    mma_fp8_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

    // ---------- 比对 ----------
    int bad = 0;
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            float got  = hD[m * 8 + n];
            float want = hRef[m * 8 + n];
            if (fabsf(got - want) > 1e-3f) {
                if (bad < 8)
                    printf("MISMATCH D[%2d][%d]: got %8.3f, want %8.3f\n",
                           m, n, got, want);
                bad++;
            }
        }

    printf("%s: %d mismatches\n", bad ? "FAIL" : "PASS", bad);

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad != 0;
}
