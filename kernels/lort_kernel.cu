#include <cuda_fp16.h>
#include <stdint.h>

// Helper to convert packed bits to half precision float
__device__ __forceinline__ half2 unpack_2bits(uint8_t pack, int shift) {
    // Extract 2 bits
    int bits_a = (pack >> shift) & 0x3;
    int bits_b = (pack >> (shift + 2)) & 0x3;
    
    // Map to float values: 0->0, 1->1, 3->-1
    // Efficient branchless logic:
    // val = (bits == 1) ? 1.0 : (bits == 3 ? -1.0 : 0.0)
    
    float a = (bits_a == 1) ? 1.0f : ((bits_a == 3) ? -1.0f : 0.0f);
    float b = (bits_b == 1) ? 1.0f : ((bits_b == 3) ? -1.0f : 0.0f);
    
    return __float22half2_rn(make_float2(a, b));
}

extern "C" __global__ void lort_fused_gemv(
    const half* __restrict__ x,       // Input [N]
    const uint8_t* __restrict__ w,    // Packed [M, N/4]
    half* __restrict__ y,             // Output [M]
    float alpha,                      // Scale
    int M, int N
) {
    // A simple GEMV kernel. In production, use Tiling / Shared Memory.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float acc = 0.0f;
    
    // Loop over columns in chunks of 4 (1 byte)
    // We use half2 (vectorized load) for speed
    for (int col_byte = 0; col_byte < N / 4; ++col_byte) {
        // Load Input (4 values = 2 x half2)
        half2 x_01 = ((half2*)x)[col_byte * 2 + 0];
        half2 x_23 = ((half2*)x)[col_byte * 2 + 1];
        
        // Load Weight (1 byte = 4 values)
        uint8_t pack = w[row * (N/4) + col_byte];
        
        // Unpack
        half2 w_01 = unpack_2bits(pack, 0);
        half2 w_23 = unpack_2bits(pack, 4);
        
        // MAC (Multiply Accumulate)
        // __hfma2 = Half Fused Multiply Add
        half2 res1 = __hmul2(x_01, w_01);
        half2 res2 = __hmul2(x_23, w_23);
        
        acc += __half2float(res1.x) + __half2float(res1.y);
        acc += __half2float(res2.x) + __half2float(res2.y);
    }

    y[row] = __float2half(acc * alpha);
}
