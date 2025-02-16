import argparse
import os
from configparser import ConfigParser

def gen_ctor_code():
    kernel_code = "\n\
#include \"ggml-bitnet.h\"\n\
#include <cstring>\n\
#define GGML_BITNET_MAX_NODES 8192\n\
static bool initialized = false;\n\
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;\n\
static size_t bitnet_tensor_extras_index = 0;\n\
static void * aligned_malloc(size_t size) {\n\
#if defined(_WIN32)\n\
    return _aligned_malloc(size, 64);\n\
#else\n\
    void * ptr = nullptr;\n\
    posix_memalign(&ptr, 64, size);\n\
    return ptr;\n\
#endif\n\
}\n\
\n\
static void aligned_free(void * ptr) {\n\
#if defined(_WIN32)\n\
    _aligned_free(ptr);\n\
#else\n\
    free(ptr);\n\
#endif\n\
}\n\
#define BK2 32\n\
#if defined __AVX2__\n\
inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi32(va, vb);\n\
    *vh = _mm256_unpackhi_epi32(va, vb);\n\
}\n\
inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi64(va, vb);\n\
    *vh = _mm256_unpackhi_epi64(va, vb);\n\
}\n\
inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));\n\
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));\n\
}\n\
inline void Transpose_8_8(\n\
    __m256i *v0,\n\
    __m256i *v1,\n\
    __m256i *v2,\n\
    __m256i *v3,\n\
    __m256i *v4,\n\
    __m256i *v5,\n\
    __m256i *v6,\n\
    __m256i *v7)\n\
{\n\
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;\n\
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;\n\
    _mm256_merge_epi32(*v0, *v1, &w0, &w1);\n\
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);\n\
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);\n\
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);\n\
    _mm256_merge_epi64(w0, w2, &x0, &x1);\n\
    _mm256_merge_epi64(w1, w3, &x2, &x3);\n\
    _mm256_merge_epi64(w4, w6, &x4, &x5);\n\
    _mm256_merge_epi64(w5, w7, &x6, &x7);\n\
    _mm256_merge_si128(x0, x4, v0, v1);\n\
    _mm256_merge_si128(x1, x5, v2, v3);\n\
    _mm256_merge_si128(x2, x6, v4, v5);\n\
    _mm256_merge_si128(x3, x7, v6, v7);\n\
}\n\
#elif defined __ARM_NEON\n\
inline void Transpose_8_8(\n\
    int8x8_t *v0,\n\
    int8x8_t *v1,\n\
    int8x8_t *v2,\n\
    int8x8_t *v3,\n\
    int8x8_t *v4,\n\
    int8x8_t *v5,\n\
    int8x8_t *v6,\n\
    int8x8_t *v7)\n\
{\n\
    int8x8x2_t q04 = vzip_s8(*v0, *v4);\n\
    int8x8x2_t q15 = vzip_s8(*v1, *v5);\n\
    int8x8x2_t q26 = vzip_s8(*v2, *v6);\n\
    int8x8x2_t q37 = vzip_s8(*v3, *v7);\n\
    int8x8x2_t q0246_0 = vzip_s8(q04.val[0], q26.val[0]);\n\
    int8x8x2_t q0246_1 = vzip_s8(q04.val[1], q26.val[1]);\n\
    int8x8x2_t q1357_0 = vzip_s8(q15.val[0], q37.val[0]);\n\
    int8x8x2_t q1357_1 = vzip_s8(q15.val[1], q37.val[1]);\n\
    int8x8x2_t q_fin_0 = vzip_s8(q0246_0.val[0], q1357_0.val[0]);\n\
    int8x8x2_t q_fin_1 = vzip_s8(q0246_0.val[1], q1357_0.val[1]);\n\
    int8x8x2_t q_fin_2 = vzip_s8(q0246_1.val[0], q1357_1.val[0]);\n\
    int8x8x2_t q_fin_3 = vzip_s8(q0246_1.val[1], q1357_1.val[1]);\n\
    *v0 = q_fin_0.val[0];\n\
    *v1 = q_fin_0.val[1];\n\
    *v2 = q_fin_1.val[0];\n\
    *v3 = q_fin_1.val[1];\n\
    *v4 = q_fin_2.val[0];\n\
    *v5 = q_fin_2.val[1];\n\
    *v6 = q_fin_3.val[0];\n\
    *v7 = q_fin_3.val[1];\n\
}\n\
#endif\n\
inline int32_t two_partial_max(void* lut_scales_, void* b_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    bitnet_float_type* b = (bitnet_float_type*)b_;\n\
#if defined __AVX2__\n\
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);\n\
    __m256 vec_b0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);\n\
    __m256 vec_b1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);\n\
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);\n\
    __m256 vec_babs0 = _mm256_andnot_ps(vec_sign, vec_b0);\n\
    __m256 vec_babs1 = _mm256_andnot_ps(vec_sign, vec_b1);\n\
    __m256 abssum = _mm256_add_ps(vec_babs0, vec_babs1);\n\
    __m128 max2 = _mm_max_ps(_mm256_extractf128_ps(abssum, 1), _mm256_castps256_ps128(abssum));\n\
    max2 = _mm_max_ps(max2, _mm_movehl_ps(max2, max2));\n\
    max2 = _mm_max_ss(max2, _mm_movehdup_ps(max2));\n\
    bitnet_float_type scales = _mm_cvtss_f32(max2) / 127;\n\
    *lut_scales = std::max(*lut_scales, scales);\n\
#elif defined __ARM_NEON\n\
    float16x8x2_t vec_bs = vld2q_f16(b);\n\
    float16x8_t abssum = vabsq_f16(vec_bs.val[0]) + vabsq_f16(vec_bs.val[1]);\n\
    float16_t scales = vmaxvq_f16(abssum) / 127;\n\
    *lut_scales = std::max(*lut_scales, scales);\n\
#endif\n\
    return 0;\n\
}\n\
inline int32_t three_partial_max(void* lut_scales_, void* b_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    bitnet_float_type* b = (bitnet_float_type*)b_;\n\
#if defined __AVX2__\n\
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);\n\
    __m256 vec_b0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);\n\
    __m256 vec_b1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);\n\
    __m256 vec_b2 = _mm256_i32gather_ps(b + 2, vec_bi, 1);\n\
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);\n\
    __m256 vec_babs0 = _mm256_andnot_ps(vec_sign, vec_b0);\n\
    __m256 vec_babs1 = _mm256_andnot_ps(vec_sign, vec_b1);\n\
    __m256 vec_babs2 = _mm256_andnot_ps(vec_sign, vec_b2);\n\
    __m256 abssum = _mm256_add_ps(_mm256_add_ps(vec_babs0, vec_babs1), vec_babs2);\n\
    __m128 max3 = _mm_max_ps(_mm256_extractf128_ps(abssum, 1), _mm256_castps256_ps128(abssum));\n\
    max3 = _mm_max_ps(max3, _mm_movehl_ps(max3, max3));\n\
    max3 = _mm_max_ss(max3, _mm_movehdup_ps(max3));\n\
    bitnet_float_type scales = _mm_cvtss_f32(max3) / 127;\n\
    *lut_scales = std::max(*lut_scales, scales);\n\
#elif defined __ARM_NEON\n\
    float16x8x3_t vec_bs = vld3q_f16(b);\n\
    float16x8_t abssum = vabsq_f16(vec_bs.val[0]) + vabsq_f16(vec_bs.val[1]) + vabsq_f16(vec_bs.val[2]);\n\
    float16_t scales = vmaxvq_f16(abssum) / 127;\n\
    *lut_scales = std::max(*lut_scales, scales);\n\
#endif\n\
    return 0;\n\
}\n\
inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    #pragma unroll\n\
    for (int i=0; i< bs; i++) {\n\
        lut_scales[i] = 0.0;\n\
    }\n\
    return 0;\n\
}\n\
template<int act_k>\n\
inline int32_t three_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256 vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);\n\
    bitnet_float_type scales = *lut_scales;\n\
    bitnet_float_type t_scales = scales ? 1.0f / scales : 0.0f;\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 24; ++k) {\n\
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);\n\
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);\n\
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);\n\
\n\
        vec_lut[15] = _mm256_setzero_ps();\n\
        vec_lut[14] = _mm256_setzero_ps();\n\
        vec_lut[13] = vec_b0;\n\
        vec_lut[13] = _mm256_add_ps(vec_lut[13], vec_b1);\n\
        vec_lut[13] = _mm256_add_ps(vec_lut[13], vec_b2);\n\
        vec_lut[12] = vec_b0;\n\
        vec_lut[12] = _mm256_add_ps(vec_lut[12], vec_b1);\n\
        vec_lut[11] = vec_b0;\n\
        vec_lut[11] = _mm256_add_ps(vec_lut[11], vec_b1);\n\
        vec_lut[11] = _mm256_sub_ps(vec_lut[11], vec_b2);\n\
        vec_lut[10] = vec_b0;\n\
        vec_lut[10] = _mm256_add_ps(vec_lut[10], vec_b2);\n\
        vec_lut[9] = vec_b0;\n\
        vec_lut[8] = vec_b0;\n\
        vec_lut[8] = _mm256_sub_ps(vec_lut[8], vec_b2);\n\
        vec_lut[7] = vec_b0;\n\
        vec_lut[7] = _mm256_sub_ps(vec_lut[7], vec_b1);\n\
        vec_lut[7] = _mm256_add_ps(vec_lut[7], vec_b2);\n\
        vec_lut[6] = vec_b0;\n\
        vec_lut[6] = _mm256_sub_ps(vec_lut[6], vec_b1);\n\
        vec_lut[5] = vec_b0;\n\
        vec_lut[5] = _mm256_sub_ps(vec_lut[5], vec_b1);\n\
        vec_lut[5] = _mm256_sub_ps(vec_lut[5], vec_b2);\n\
        vec_lut[4] = vec_b1;\n\
        vec_lut[4] = _mm256_add_ps(vec_lut[4], vec_b2);\n\
        vec_lut[3] = vec_b1;\n\
        vec_lut[2] = vec_b1;\n\
        vec_lut[2] = _mm256_sub_ps(vec_lut[2], vec_b2);\n\
        vec_lut[1] = vec_b2;\n\
        vec_lut[0] = _mm256_setzero_ps();\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 14; ++g) {\n\
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));\n\
        }\n\
        __m256i ix[16];\n\
        for (int g = 0; g < 14; ++g) {\n\
            ix[g] = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        }\n\
        __m256i shuffle_mask = _mm256_set_epi8(\n\
                                               0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04,\n\
                                               0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00,\n\
                                               0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04,\n\
                                               0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00\n\
                                               );\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_packs_epi16(ix[g], ix[g]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            _mm_storeu_si128(reinterpret_cast<__m128i*>(qlut_i8 + k * 128 + g * 16 + 0), _mm256_castsi256_si128(ix[g]));\n\
        }\n\
    }\n\
\n\
    *lut_scales = scales;\n\
#elif defined __ARM_NEON\n\
    float16x8_t vec_lut[16];\n\
    float16_t scales = *lut_scales;\n\
    float16_t t_scales = scales ? 1.0 / scales : 0.0;\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 24; ++k) {\n\
        float16x8x3_t vec_bs = vld3q_f16(b + k * 24);\n\
        vec_lut[15] = vdupq_n_f16(0);\n\
        vec_lut[14] = vdupq_n_f16(0);\n\
        vec_lut[13] = vec_bs.val[0] + vec_bs.val[1] + vec_bs.val[2];\n\
        vec_lut[12] = vec_bs.val[0] + vec_bs.val[1];\n\
        vec_lut[11] = vec_bs.val[0] + vec_bs.val[1] - vec_bs.val[2];\n\
        vec_lut[10] = vec_bs.val[0] + vec_bs.val[2];\n\
        vec_lut[9] = vec_bs.val[0];\n\
        vec_lut[8] = vec_bs.val[0] - vec_bs.val[2];\n\
        vec_lut[7] = vec_bs.val[0] - vec_bs.val[1] + vec_bs.val[2];\n\
        vec_lut[6] = vec_bs.val[0] - vec_bs.val[1];\n\
        vec_lut[5] = vec_bs.val[0] - vec_bs.val[1] - vec_bs.val[2];\n\
        vec_lut[4] = vec_bs.val[1] + vec_bs.val[2];\n\
        vec_lut[3] = vec_bs.val[1];\n\
        vec_lut[2] = vec_bs.val[1] - vec_bs.val[2];\n\
        vec_lut[1] = vec_bs.val[2];\n\
        vec_lut[0] = vdupq_n_f16(0);\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 14; ++g) {\n\
            vec_lut[g] = vmulq_n_f16(vec_lut[g], t_scales);\n\
        }\n\
\n\
        int8x8_t vec_qlut[16];\n\
#pragma unroll\n\
        for (int g = 0; g < 14; ++g) {\n\
            vec_qlut[g] = vqmovn_s16(vcvtnq_s16_f16(vec_lut[g]));\n\
        }\n\
        Transpose_8_8(&(vec_qlut[0]), &(vec_qlut[1]), &(vec_qlut[2]), &(vec_qlut[3]),\n\
                      &(vec_qlut[4]), &(vec_qlut[5]), &(vec_qlut[6]), &(vec_qlut[7]));\n\
        Transpose_8_8(&(vec_qlut[8]), &(vec_qlut[9]), &(vec_qlut[10]), &(vec_qlut[11]),\n\
                      &(vec_qlut[12]), &(vec_qlut[13]), &(vec_qlut[14]), &(vec_qlut[15]));\n\
\n\
#pragma unroll\n\
        for (int idx = 0; idx < 8; idx++) {\n\
            vst1_s8(qlut + k * 16 * 8 + idx * 16 + 0 * 8, vec_qlut[idx]);\n\
            vst1_s8(qlut + k * 16 * 8 + idx * 16 + 1 * 8, vec_qlut[idx + 8]);\n\
        }\n\
    }\n\
#endif\n\
    return 0;\n\
}\n\
\n\
template<int act_k>\n\
inline int32_t two_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256 vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);\n\
    bitnet_float_type scales = *lut_scales;\n\
    bitnet_float_type t_scales = scales ? 1.0f / scales : 0.0f;\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 16; ++k) {\n\
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);\n\
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);\n\
        vec_lut[0] = _mm256_setzero_ps();\n\
        vec_lut[0] = _mm256_sub_ps(vec_lut[0], vec_b0);\n\
        vec_lut[0] = _mm256_sub_ps(vec_lut[0], vec_b1);\n\
        vec_lut[1] = _mm256_setzero_ps();\n\
        vec_lut[1] = _mm256_sub_ps(vec_lut[1], vec_b0);\n\
        vec_lut[2] = _mm256_setzero_ps();\n\
        vec_lut[2] = _mm256_sub_ps(vec_lut[2], vec_b0);\n\
        vec_lut[2] = _mm256_add_ps(vec_lut[2], vec_b1);\n\
        vec_lut[3] = _mm256_setzero_ps();\n\
        vec_lut[3] = _mm256_sub_ps(vec_lut[3], vec_b1);\n\
        vec_lut[4] = _mm256_setzero_ps();\n\
        vec_lut[5] = vec_b1;\n\
        vec_lut[6] = vec_b0;\n\
        vec_lut[6] = _mm256_sub_ps(vec_lut[6], vec_b1);\n\
        vec_lut[7] = vec_b0;\n\
        vec_lut[8] = vec_b0;\n\
        vec_lut[8] = _mm256_add_ps(vec_lut[8], vec_b1);\n\
        vec_lut[9] = _mm256_setzero_ps();\n\
        vec_lut[10] = _mm256_setzero_ps();\n\
        vec_lut[11] = _mm256_setzero_ps();\n\
        vec_lut[12] = _mm256_setzero_ps();\n\
        vec_lut[13] = _mm256_setzero_ps();\n\
        vec_lut[14] = _mm256_setzero_ps();\n\
        vec_lut[15] = _mm256_setzero_ps();\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 9; ++g) {\n\
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));\n\
        }\n\
        __m256i ix[16];\n\
#pragma unroll\n\
        for (int g = 0; g < 9; ++g) {\n\
            ix[g] = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        }\n\
\n\
        __m256i shuffle_mask = _mm256_set_epi8(\n\
                                               0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04,\n\
                                               0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00,\n\
                                               0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04,\n\
                                               0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00\n\
                                               );\n\
\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_packs_epi16(ix[g], ix[g]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            _mm_storeu_si128(reinterpret_cast<__m128i*>(qlut_i8 + k * 128 + g * 16 + 0), _mm256_castsi256_si128(ix[g]));\n\
        }\n\
    }\n\
    *lut_scales = scales;\n\
#elif defined __ARM_NEON\n\
    float16x8_t vec_lut[16];\n\
    float16_t scales = *lut_scales;\n\
    float16_t t_scales = scales ? 1.0 / scales : 0.0;\n\
\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 16; ++k) {\n\
        float16x8x2_t vec_bs = vld2q_f16(b + k * 16);\n\
        vec_lut[15] = vdupq_n_f16(0);\n\
        vec_lut[14] = vdupq_n_f16(0);\n\
        vec_lut[13] = vdupq_n_f16(0);\n\
        vec_lut[12] = vdupq_n_f16(0);\n\
        vec_lut[11] = vdupq_n_f16(0);\n\
        vec_lut[10] = vdupq_n_f16(0);\n\
        vec_lut[9] = vdupq_n_f16(0);\n\
        vec_lut[8] = vec_bs.val[0] + vec_bs.val[1];\n\
        vec_lut[7] = vec_bs.val[0];\n\
        vec_lut[6] = vec_bs.val[0] - vec_bs.val[1];\n\
        vec_lut[5] = vec_bs.val[1];\n\
        vec_lut[4] = vdupq_n_f16(0);\n\
        vec_lut[3] = -vec_bs.val[1];\n\
        vec_lut[2] = -vec_bs.val[0] + vec_bs.val[1];\n\
        vec_lut[1] = -vec_bs.val[0];\n\
        vec_lut[0] = -vec_bs.val[0] - vec_bs.val[1];\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            vec_lut[g] = vmulq_n_f16(vec_lut[g], t_scales);\n\
        }\n\
\n\
        int8x8_t vec_qlut[16];\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            vec_qlut[g] = vqmovn_s16(vcvtnq_s16_f16(vec_lut[g]));\n\
        }\n\
        Transpose_8_8(&(vec_qlut[0]), &(vec_qlut[1]), &(vec_qlut[2]), &(vec_qlut[3]),\n\
                      &(vec_qlut[4]), &(vec_qlut[5]), &(vec_qlut[6]), &(vec_qlut[7]));\n\
        Transpose_8_8(&(vec_qlut[8]), &(vec_qlut[9]), &(vec_qlut[10]), &(vec_qlut[11]),\n\
                      &(vec_qlut[12]), &(vec_qlut[13]), &(vec_qlut[14]), &(vec_qlut[15]));\n\
\n\
#pragma unroll\n\
        for (int idx = 0; idx < 8; idx++) {\n\
            vst1_s8(qlut + k * 16 * 8 + idx * 16 + 0 * 8, vec_qlut[idx]);\n\
            vst1_s8(qlut + k * 16 * 8 + idx * 16 + 1 * 8, vec_qlut[idx + 8]);\n\
        }\n\
    }\n\
#endif\n\
    return 0;\n\
}\n\
static bool is_type_supported(enum ggml_type type) {\n\
    if (type == GGML_TYPE_Q4_0 ||\n\
        type == GGML_TYPE_TL2) {\n\
        return true;\n\
    } else {\n\
        return false;\n\
    }\n\
}\n\
"
    return kernel_code

def gen_tbl_impl(pre, BM, BK, bm, k_list):

    kernel_code = "\
\n\
#define BM{0} {1}\n\
#define BBK{0} {2}\n\
template<int batch_size, int K3>\n\
inline void three_tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {{\n\
".format(pre, BM, BK)

    if bm == 16:
        kernel_code = "".join([kernel_code, "\
#ifdef __AVX2__\n\
    const int KK = BBK{0}/ 3;\n\
    for (int i = 0; i < BM{0}; i += 16) {{\n\
        __m256i vec_c0 = _mm256_setzero_si256();\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 16; k++) {{\n\
            __m256i vec_sign = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + k * 32));\n\
            __m256i vec_k_top_256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 0 * 64));\n\
            __m256i vec_k_bot_256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 0 * 64 + 32));\n\
            __m256i vec_a_0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 128 + 0 * 32));\n\
            __m256i vec_a_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_0 = _mm256_and_si256(vec_a_0, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 2 * 0), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_0 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_0, vec_a_top_0), vec_sign_top_0), vec_sign_top_0);\n\
            __m256i vec_sign_bot_0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 6 - 2 * 0), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_0 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_0, vec_a_bot_0), vec_sign_bot_0), vec_sign_bot_0);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_0)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_0)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_0, 1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_0, 1)));\n\
            __m256i vec_k_top_256_1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 1 * 64));\n\
            __m256i vec_k_bot_256_1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 1 * 64 + 32));\n\
            __m256i vec_a_1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 128 + 1 * 32));\n\
            __m256i vec_a_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_1 = _mm256_and_si256(vec_a_1, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 2 * 1), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_1 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_1, vec_a_top_1), vec_sign_top_1), vec_sign_top_1);\n\
            __m256i vec_sign_bot_1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 6 - 2 * 1), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_1 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_1, vec_a_bot_1), vec_sign_bot_1), vec_sign_bot_1);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_1, 1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_1, 1)));\n\
            __m256i vec_k_top_256_2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 2 * 64));\n\
            __m256i vec_k_bot_256_2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 2 * 64 + 32));\n\
            __m256i vec_a_2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 128 + 2 * 32));\n\
            __m256i vec_a_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_2 = _mm256_and_si256(vec_a_2, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 2 * 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_2 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_2, vec_a_top_2), vec_sign_top_2), vec_sign_top_2);\n\
            __m256i vec_sign_bot_2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 6 - 2 * 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_2 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_2, vec_a_bot_2), vec_sign_bot_2), vec_sign_bot_2);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_2)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_2)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_2, 1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_2, 1)));\n\
            __m256i vec_k_top_256_3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 3 * 64));\n\
            __m256i vec_k_bot_256_3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 256 * k + 3 * 64 + 32));\n\
            __m256i vec_a_3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 128 + 3 * 32));\n\
            __m256i vec_a_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_3 = _mm256_and_si256(vec_a_3, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 3 * 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_3 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_3, vec_a_top_3), vec_sign_top_3), vec_sign_top_3);\n\
            __m256i vec_sign_bot_3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 6 - 3 * 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_3 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_3, vec_a_bot_3), vec_sign_bot_3), vec_sign_bot_3);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_3)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_3)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_3, 1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_3, 1)));\n\
        }}\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8));\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8), vec_gc1);\n\
    }}\n".format(pre)])

        kernel_code = "".join([kernel_code, "\
#elif defined __ARM_NEON\n\
    const int KK = BBK{0} / 3;\n\
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i += 16) {{\n\
        int16x8_t vec_c0 = vdupq_n_s16(0);\n\
        int16x8_t vec_c1 = vdupq_n_s16(0);\n\
#pragma unroll \n\
        for (int k = 0; k < KK / 16; k++) {{\n\
            uint8x16_t vec_sign_left = vmvnq_s8(vld1q_u8(sign + i * KK / 8 + k * 32));\n\
            uint8x16_t vec_sign_right = vmvnq_u8(vld1q_u8(sign + i * KK / 8 + k * 32 + 16));\n".format(pre)])

        for i in range(4):
            kernel_code = "".join([kernel_code, "\
            int8x16_t vec_k_left_left_{0} = vld1q_s8(lut + 256 * k + {0} * 64);\n\
            int8x16_t vec_k_left_right_{0} = vld1q_s8(lut + 256 * k + {0} * 64 + 16);\n\
            int8x16_t vec_k_right_left_{0} = vld1q_s8(lut + 256 * k + {0} * 64 + 32);\n\
            int8x16_t vec_k_right_right_{0} = vld1q_s8(lut + 256 * k + {0} * 64 + 48);\n\
            uint8x16_t vec_sign_left_left_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_left, 2 * {0}));\n\
            uint8x16_t vec_sign_left_right_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_left, 2 * {0} + 1));\n\
            uint8x16_t vec_sign_right_left_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_right, 2 * {0}));\n\
            uint8x16_t vec_sign_right_right_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_right, 2 * {0} + 1));\n\
            uint8x16_t vec_a_left_{0} = vld1q_u8(a + i * KK / 2 + k * 128 + {0} * 32);\n\
            uint8x16_t vec_a_right_{0} = vld1q_u8(a + i * KK / 2 + k * 128 + {0} * 32 + 16);\n\
            uint8x16_t vec_a_left_left_{0} = vshrq_n_u8(vec_a_left_{0}, 4);\n\
            uint8x16_t vec_a_left_right_{0} = vandq_u8(vec_a_left_{0}, vec_mask);\n\
            uint8x16_t vec_a_right_left_{0} = vshrq_n_u8(vec_a_right_{0}, 4);\n\
            uint8x16_t vec_a_right_right_{0} = vandq_u8(vec_a_right_{0}, vec_mask);\n\
            int8x16_t vec_v_top_left_tmp_{0} = vqtbl1q_s8(vec_k_left_left_{0}, vec_a_left_left_{0});\n\
            int8x16_t vec_v_bot_left_tmp_{0} = vqtbl1q_s8(vec_k_left_right_{0}, vec_a_right_left_{0});\n\
            int8x16_t vec_v_top_right_tmp_{0} = vqtbl1q_s8(vec_k_right_left_{0}, vec_a_left_right_{0});\n\
            int8x16_t vec_v_bot_right_tmp_{0} = vqtbl1q_s8(vec_k_right_right_{0}, vec_a_right_right_{0});\n\
            vec_v_top_left_tmp_{0} = vbslq_s8(vec_sign_left_left_{0}, vnegq_s8(vec_v_top_left_tmp_{0}), vec_v_top_left_tmp_{0});\n\
            vec_v_bot_left_tmp_{0} = vbslq_s8(vec_sign_right_left_{0}, vnegq_s8(vec_v_bot_left_tmp_{0}), vec_v_bot_left_tmp_{0});\n\
            vec_v_top_right_tmp_{0} = vbslq_s8(vec_sign_left_right_{0}, vnegq_s8(vec_v_top_right_tmp_{0}), vec_v_top_right_tmp_{0});\n\
            vec_v_bot_right_tmp_{0} = vbslq_s8(vec_sign_right_right_{0}, vnegq_s8(vec_v_bot_right_tmp_{0}), vec_v_bot_right_tmp_{0});\n\
            int16x8_t vec_v_top_left_high_{0} = vmovl_high_s8(vec_v_top_left_tmp_{0});\n\
            int16x8_t vec_v_top_left_bot_{0} = vmovl_s8(vget_low_s8(vec_v_top_left_tmp_{0}));\n\
            int16x8_t vec_v_top_right_high_{0} = vmovl_high_s8(vec_v_top_right_tmp_{0});\n\
            int16x8_t vec_v_top_right_bot_{0} = vmovl_s8(vget_low_s8(vec_v_top_right_tmp_{0}));\n\
            int16x8_t vec_v_bot_left_high_{0} = vmovl_high_s8(vec_v_bot_left_tmp_{0});\n\
            int16x8_t vec_v_bot_left_bot_{0} = vmovl_s8(vget_low_s8(vec_v_bot_left_tmp_{0}));\n\
            int16x8_t vec_v_bot_right_high_{0} = vmovl_high_s8(vec_v_bot_right_tmp_{0});\n\
            int16x8_t vec_v_bot_right_bot_{0} = vmovl_s8(vget_low_s8(vec_v_bot_right_tmp_{0}));\n\
            vec_c0 += vec_v_top_left_bot_{0};\n\
            vec_c0 += vec_v_top_right_bot_{0};\n\
            vec_c0 += vec_v_bot_left_bot_{0};\n\
            vec_c0 += vec_v_bot_right_bot_{0};\n\
            vec_c1 += vec_v_top_left_high_{0};\n\
            vec_c1 += vec_v_top_right_high_{0};\n\
            vec_c1 += vec_v_bot_left_high_{0};\n\
            vec_c1 += vec_v_bot_right_high_{0};\n".format(i)])

        kernel_code = "".join([kernel_code, "\
        }\n\
        int32x4_t vec_v_1 = vmovl_high_s16(vec_c0);\n\
        int32x4_t vec_v_0 = vmovl_s16(vget_low_s16(vec_c0));\n\
        int32x4_t vec_v_3 = vmovl_high_s16(vec_c1);\n\
        int32x4_t vec_v_2 = vmovl_s16(vget_low_s16(vec_c1));\n\
        vst1q_s32(c + i,      vld1q_s32(c + i     ) + vec_v_0);\n\
        vst1q_s32(c + i + 4,  vld1q_s32(c + i + 4 ) + vec_v_1);\n\
        vst1q_s32(c + i + 8,  vld1q_s32(c + i + 8 ) + vec_v_2);\n\
        vst1q_s32(c + i + 12, vld1q_s32(c + i + 12) + vec_v_3);\n\
    }\n\
#endif\n\
}\n"])
    elif bm == 32:
        kernel_code = "".join([kernel_code, "\
#ifdef __AVX2__\n\
    const int KK = BBK{0} / 3;\n\
    for (int i = 0; i < BM{0}; i += 32) {{\n\
        __m256i vec_c0 = _mm256_set1_epi16(0);\n\
        __m256i vec_c1 = _mm256_set1_epi16(0);\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 8; k++) {{\n\
            __m256i vec_sign = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + k * 32));\n\
            __m128i vec_k_top_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 0 * 32 + 0));\n\
            __m256i vec_k_top_256_0 = _mm256_set_m128i(vec_k_top_0, vec_k_top_0);\n\
            __m128i vec_k_bot_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 0 * 32 + 16));\n\
            __m256i vec_k_bot_256_0 = _mm256_set_m128i(vec_k_bot_0, vec_k_bot_0);\n\
            __m256i vec_a_0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 32 * 4 + 0 * 32));\n\
            __m256i vec_a_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_0 = _mm256_and_si256(vec_a_0, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 0), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_0 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_0, vec_a_top_0), vec_sign_top_0), vec_sign_top_0);\n\
            __m256i vec_sign_bot_0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 3 - 0), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_0 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_0, vec_a_bot_0), vec_sign_bot_0), vec_sign_bot_0);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_0)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_0)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_0, 1)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_0, 1)));\n\
            __m128i vec_k_top_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 1 * 32 + 0));\n\
            __m256i vec_k_top_256_1 = _mm256_set_m128i(vec_k_top_1, vec_k_top_1);\n\
            __m128i vec_k_bot_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 1 * 32 + 16));\n\
            __m256i vec_k_bot_256_1 = _mm256_set_m128i(vec_k_bot_1, vec_k_bot_1);\n\
            __m256i vec_a_1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 32 * 4 + 1 * 32));\n\
            __m256i vec_a_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_1 = _mm256_and_si256(vec_a_1, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 1), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_1 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_1, vec_a_top_1), vec_sign_top_1), vec_sign_top_1);\n\
            __m256i vec_sign_bot_1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 3 - 1), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_1 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_1, vec_a_bot_1), vec_sign_bot_1), vec_sign_bot_1);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_1)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_1, 1)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_1, 1)));\n\
            __m128i vec_k_top_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 2 * 32 + 0));\n\
            __m256i vec_k_top_256_2 = _mm256_set_m128i(vec_k_top_2, vec_k_top_2);\n\
            __m128i vec_k_bot_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 2 * 32 + 16));\n\
            __m256i vec_k_bot_256_2 = _mm256_set_m128i(vec_k_bot_2, vec_k_bot_2);\n\
            __m256i vec_a_2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 32 * 4 + 2 * 32));\n\
            __m256i vec_a_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_2 = _mm256_and_si256(vec_a_2, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_2 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_2, vec_a_top_2), vec_sign_top_2), vec_sign_top_2);\n\
            __m256i vec_sign_bot_2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 3 - 2), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_2 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_2, vec_a_bot_2), vec_sign_bot_2), vec_sign_bot_2);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_2)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_2)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_2, 1)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_2, 1)));\n\
            __m128i vec_k_top_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 3 * 32 + 0));\n\
            __m256i vec_k_top_256_3 = _mm256_set_m128i(vec_k_top_3, vec_k_top_3);\n\
            __m128i vec_k_bot_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 128 * k + 3 * 32 + 16));\n\
            __m256i vec_k_bot_256_3 = _mm256_set_m128i(vec_k_bot_3, vec_k_bot_3);\n\
            __m256i vec_a_3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + k * 32 * 4 + 3 * 32));\n\
            __m256i vec_a_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), _mm256_set1_epi8(0x0f));\n\
            __m256i vec_a_bot_3 = _mm256_and_si256(vec_a_3, _mm256_set1_epi8(0x0f));\n\
            __m256i vec_sign_top_3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 7 - 3), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_top_3 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_top_256_3, vec_a_top_3), vec_sign_top_3), vec_sign_top_3);\n\
            __m256i vec_sign_bot_3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(vec_sign, 3 - 3), _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));\n\
            __m256i vec_v_bot_3 = _mm256_xor_si256(_mm256_add_epi8(_mm256_shuffle_epi8(vec_k_bot_256_3, vec_a_bot_3), vec_sign_bot_3), vec_sign_bot_3);\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot_3)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top_3)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot_3, 1)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top_3, 1)));\n\
        }}\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8));\n\
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16));\n\
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24));\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n\
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8), vec_gc1);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16), vec_gc2);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24), vec_gc3);\n\
    }}\n".format(pre)])

        kernel_code = "".join([kernel_code, "\
#elif defined __ARM_NEON\n\
    const int KK = BBK{0} / 3;\n\
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i += 32) {{\n\
        int16x8_t vec_c0 = vdupq_n_s16(0);\n\
        int16x8_t vec_c1 = vdupq_n_s16(0);\n\
        int16x8_t vec_c2 = vdupq_n_s16(0);\n\
        int16x8_t vec_c3 = vdupq_n_s16(0);\n\
#pragma unroll \n\
        for (int k = 0; k < KK / 8; k++) {{\n\
            uint8x16_t vec_sign_left = vmvnq_s8(vld1q_u8(sign + i * KK / 8 + k * 32));\n\
            uint8x16_t vec_sign_right = vmvnq_u8(vld1q_u8(sign + i * KK / 8 + k * 32 + 16));\n".format(pre)])

        for i in range(4):
            kernel_code = "".join([kernel_code, "\
            int8x16_t vec_k_left_{0} = vld1q_s8(lut + 128 * k + {0} * 32);\n\
            int8x16_t vec_k_right_{0} = vld1q_s8(lut + 128 * k + {0} * 32 + 16);\n\
            uint8x16_t vec_sign_left_left_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_left, {0}));\n\
            uint8x16_t vec_sign_left_right_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_left, {0} + 4));\n\
            uint8x16_t vec_sign_right_left_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_right, {0}));\n\
            uint8x16_t vec_sign_right_right_{0} = vcltzq_s8(vshlq_n_u8(vec_sign_right, {0} + 4));\n\
            uint8x16_t vec_a_left_{0} = vld1q_u8(a + i * KK / 2 + k * 128 + {0} * 32);\n\
            uint8x16_t vec_a_right_{0} = vld1q_u8(a + i * KK / 2 + k * 128 + {0} * 32 + 16);\n\
            uint8x16_t vec_a_left_left_{0} = vshrq_n_u8(vec_a_left_{0}, 4);\n\
            uint8x16_t vec_a_left_right_{0} = vandq_u8(vec_a_left_{0}, vec_mask);\n\
            uint8x16_t vec_a_right_left_{0} = vshrq_n_u8(vec_a_right_{0}, 4);\n\
            uint8x16_t vec_a_right_right_{0} = vandq_u8(vec_a_right_{0}, vec_mask);\n\
            int8x16_t vec_v_top_left_tmp_{0} = vqtbl1q_s8(vec_k_left_{0}, vec_a_left_left_{0});\n\
            int8x16_t vec_v_bot_left_tmp_{0} = vqtbl1q_s8(vec_k_left_{0}, vec_a_right_left_{0});\n\
            int8x16_t vec_v_top_right_tmp_{0} = vqtbl1q_s8(vec_k_right_{0}, vec_a_left_right_{0});\n\
            int8x16_t vec_v_bot_right_tmp_{0} = vqtbl1q_s8(vec_k_right_{0}, vec_a_right_right_{0});\n\
            vec_v_top_left_tmp_{0} = vbslq_s8(vec_sign_left_left_{0}, vnegq_s8(vec_v_top_left_tmp_{0}), vec_v_top_left_tmp_{0});\n\
            vec_v_bot_left_tmp_{0} = vbslq_s8(vec_sign_right_left_{0}, vnegq_s8(vec_v_bot_left_tmp_{0}), vec_v_bot_left_tmp_{0});\n\
            vec_v_top_right_tmp_{0} = vbslq_s8(vec_sign_left_right_{0}, vnegq_s8(vec_v_top_right_tmp_{0}), vec_v_top_right_tmp_{0});\n\
            vec_v_bot_right_tmp_{0} = vbslq_s8(vec_sign_right_right_{0}, vnegq_s8(vec_v_bot_right_tmp_{0}), vec_v_bot_right_tmp_{0});\n\
            int16x8_t vec_v_top_left_high_{0} = vmovl_high_s8(vec_v_top_left_tmp_{0});\n\
            int16x8_t vec_v_top_left_bot_{0} = vmovl_s8(vget_low_s8(vec_v_top_left_tmp_{0}));\n\
            int16x8_t vec_v_top_right_high_{0} = vmovl_high_s8(vec_v_top_right_tmp_{0});\n\
            int16x8_t vec_v_top_right_bot_{0} = vmovl_s8(vget_low_s8(vec_v_top_right_tmp_{0}));\n\
            int16x8_t vec_v_bot_left_high_{0} = vmovl_high_s8(vec_v_bot_left_tmp_{0});\n\
            int16x8_t vec_v_bot_left_bot_{0} = vmovl_s8(vget_low_s8(vec_v_bot_left_tmp_{0}));\n\
            int16x8_t vec_v_bot_right_high_{0} = vmovl_high_s8(vec_v_bot_right_tmp_{0});\n\
            int16x8_t vec_v_bot_right_bot_{0} = vmovl_s8(vget_low_s8(vec_v_bot_right_tmp_{0}));\n\
            vec_c0 += vec_v_top_left_bot_{0};\n\
            vec_c0 += vec_v_top_right_bot_{0};\n\
            vec_c1 += vec_v_bot_left_bot_{0};\n\
            vec_c1 += vec_v_bot_right_bot_{0};\n\
            vec_c2 += vec_v_top_left_high_{0};\n\
            vec_c2 += vec_v_top_right_high_{0};\n\
            vec_c3 += vec_v_bot_left_high_{0};\n\
            vec_c3 += vec_v_bot_right_high_{0};\n".format(i)])

        kernel_code = "".join([kernel_code, "\
        }\n\
        int32x4_t vec_v_1 = vmovl_high_s16(vec_c0);\n\
        int32x4_t vec_v_0 = vmovl_s16(vget_low_s16(vec_c0));\n\
        int32x4_t vec_v_3 = vmovl_high_s16(vec_c1);\n\
        int32x4_t vec_v_2 = vmovl_s16(vget_low_s16(vec_c1));\n\
        int32x4_t vec_v_5 = vmovl_high_s16(vec_c2);\n\
        int32x4_t vec_v_4 = vmovl_s16(vget_low_s16(vec_c2));\n\
        int32x4_t vec_v_7 = vmovl_high_s16(vec_c3);\n\
        int32x4_t vec_v_6 = vmovl_s16(vget_low_s16(vec_c3));\n\
\n\
        vst1q_s32(c + i,      vld1q_s32(c + i     ) + vec_v_0);\n\
        vst1q_s32(c + i + 4,  vld1q_s32(c + i + 4 ) + vec_v_1);\n\
        vst1q_s32(c + i + 8,  vld1q_s32(c + i + 8 ) + vec_v_4);\n\
        vst1q_s32(c + i + 12, vld1q_s32(c + i + 12) + vec_v_5);\n\
        vst1q_s32(c + i + 16, vld1q_s32(c + i + 16) + vec_v_2);\n\
        vst1q_s32(c + i + 20, vld1q_s32(c + i + 20) + vec_v_3);\n\
        vst1q_s32(c + i + 24, vld1q_s32(c + i + 24) + vec_v_6);\n\
        vst1q_s32(c + i + 28, vld1q_s32(c + i + 28) + vec_v_7);\n\
    }\n\
#endif\n\
}\n"])

    kernel_code = "".join([kernel_code, "\
\n\
template<int batch_size, int K2>\n\
inline int32_t two_tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a) {{\n\
#ifdef __AVX2__\n\
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);\n\
    const __m256i vec_sub  = _mm256_set1_epi8(0x01);\n\
    const int KK = 16;\n\
    __m256i vec_lut[KK];\n\
    for (int k = 0; k < KK; k++) {{\n\
        __m128i vec_k = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + 8 * k));\n\
        vec_lut[k] = _mm256_set_m128i(vec_k, vec_k);\n\
    }}\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0} / 2; i += 16) {{\n\
        __m256i vec_c0 = _mm256_set1_epi16(0);\n\
        __m256i vec_c1 = _mm256_set1_epi16(0);\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 2; k++) {{\n\
            __m256i vec_as = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK + k * 32));\n\
            __m256i vec_v_bot = _mm256_shuffle_epi8(vec_lut[2 * k + 1], _mm256_and_si256(vec_as, vec_mask));\n\
            __m256i vec_v_top = _mm256_shuffle_epi8(vec_lut[2 * k], _mm256_and_si256(_mm256_srli_epi16(vec_as, 4), vec_mask));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_bot)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_bot, 1)));\n\
            vec_c0 = _mm256_add_epi16(vec_c0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_v_top)));\n\
            vec_c1 = _mm256_add_epi16(vec_c1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_v_top, 1)));\n\
        }}\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8));\n\
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16));\n\
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24));\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n\
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8), vec_gc1);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16), vec_gc2);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24), vec_gc3);\n\
    }}\n\
#elif defined __ARM_NEON\n\
    const int KK = 16;\n\
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);\n\
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);\n\
    int8x16_t vec_lut[KK];\n\
#pragma unroll\n\
    for (int k = 0; k < KK; k++) {{\n\
        vec_lut[k] = vld1q_s8(lut + k * 16);\n\
    }}\n\
    for (int i = 0; i < BM{0} / 2; i += 16) {{\n\
        int16x8_t vec_c0 = vdupq_n_s16(0);\n\
        int16x8_t vec_c1 = vdupq_n_s16(0);\n\
        int16x8_t vec_c2 = vdupq_n_s16(0);\n\
        int16x8_t vec_c3 = vdupq_n_s16(0);\n\
        for (int k = 0; k < KK / 2; k++) {{\n\
            uint8x16_t vec_a_top = vld1q_u8(a + i * KK + k * 32);\n\
            uint8x16_t vec_a_bot = vld1q_u8(a + i * KK + k * 32 + 16);\n\
            uint8x16_t vec_a_top_left = vshrq_n_u8(vec_a_top, 4);\n\
            uint8x16_t vec_a_top_right = vandq_u8(vec_a_top, vec_mask);\n\
            uint8x16_t vec_a_bot_left = vshrq_n_u8(vec_a_bot, 4);\n\
            uint8x16_t vec_a_bot_right = vandq_u8(vec_a_bot, vec_mask);\n\
            int8x16_t vec_v_top_left_tmp = vqtbl1q_s8(vec_lut[2 * k], vec_a_top_left);\n\
            int8x16_t vec_v_top_right_tmp = vqtbl1q_s8(vec_lut[2 * k + 1], vec_a_top_right);\n\
            int8x16_t vec_v_bot_left_tmp = vqtbl1q_s8(vec_lut[2 * k], vec_a_bot_left);\n\
            int8x16_t vec_v_bot_right_tmp = vqtbl1q_s8(vec_lut[2 * k + 1], vec_a_bot_right);\n\
            int16x8_t vec_v_top_left_high = vmovl_high_s8(vec_v_top_left_tmp);\n\
            int16x8_t vec_v_top_left_bot = vmovl_s8(vget_low_s8(vec_v_top_left_tmp));\n\
            int16x8_t vec_v_top_right_high = vmovl_high_s8(vec_v_top_right_tmp);\n\
            int16x8_t vec_v_top_right_bot = vmovl_s8(vget_low_s8(vec_v_top_right_tmp));\n\
            int16x8_t vec_v_bot_left_high = vmovl_high_s8(vec_v_bot_left_tmp);\n\
            int16x8_t vec_v_bot_left_bot = vmovl_s8(vget_low_s8(vec_v_bot_left_tmp));\n\
            int16x8_t vec_v_bot_right_high = vmovl_high_s8(vec_v_bot_right_tmp);\n\
            int16x8_t vec_v_bot_right_bot = vmovl_s8(vget_low_s8(vec_v_bot_right_tmp));\n\
            vec_c0 += vec_v_top_left_bot;\n\
            vec_c0 += vec_v_top_right_bot;\n\
            vec_c1 += vec_v_top_left_high;\n\
            vec_c1 += vec_v_top_right_high;\n\
            vec_c2 += vec_v_bot_left_bot;\n\
            vec_c2 += vec_v_bot_right_bot;\n\
            vec_c3 += vec_v_bot_left_high;\n\
            vec_c3 += vec_v_bot_right_high;\n\
        }}\n\
        int32x4_t vec_v_1 = vmovl_high_s16(vec_c0);\n\
        int32x4_t vec_v_0 = vmovl_s16(vget_low_s16(vec_c0));\n\
        int32x4_t vec_v_3 = vmovl_high_s16(vec_c1);\n\
        int32x4_t vec_v_2 = vmovl_s16(vget_low_s16(vec_c1));\n\
        int32x4_t vec_v_5 = vmovl_high_s16(vec_c2);\n\
        int32x4_t vec_v_4 = vmovl_s16(vget_low_s16(vec_c2));\n\
        int32x4_t vec_v_7 = vmovl_high_s16(vec_c3);\n\
        int32x4_t vec_v_6 = vmovl_s16(vget_low_s16(vec_c3));\n\
        vst1q_s32(c + i * 2,      vld1q_s32(c + i * 2     ) + vec_v_0);\n\
        vst1q_s32(c + i * 2 + 4,  vld1q_s32(c + i * 2 + 4 ) + vec_v_1);\n\
        vst1q_s32(c + i * 2 + 8,  vld1q_s32(c + i * 2 + 8 ) + vec_v_2);\n\
        vst1q_s32(c + i * 2 + 12, vld1q_s32(c + i * 2 + 12) + vec_v_3);\n\
        vst1q_s32(c + i * 2 + 16, vld1q_s32(c + i * 2 + 16) + vec_v_4);\n\
        vst1q_s32(c + i * 2 + 20, vld1q_s32(c + i * 2 + 20) + vec_v_5);\n\
        vst1q_s32(c + i * 2 + 24, vld1q_s32(c + i * 2 + 24) + vec_v_6);\n\
        vst1q_s32(c + i * 2 + 28, vld1q_s32(c + i * 2 + 28) + vec_v_7);\n\
    }}\n\
#endif\n\
    return 0;\n\
}};\n\
\n\
template<int BATCH_SIZE>\n\
int32_t three_qgemm_lut_{0}(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    alignas(32) uint32_t CBits[BATCH_SIZE * BM{0}];\n\
    memset(&(CBits[0]), 0, BATCH_SIZE * BM{0} * sizeof(int32_t));\n\
#pragma unroll\n\
    for (int32_t k_outer = 0; k_outer < {1} / BBK{0}; ++k_outer) {{\n\
        three_tbl_impl_{0}<BATCH_SIZE, {1}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK{0} / 3 * 16)])), (&(((uint8_t*)A)[(k_outer * BBK{0} / 3 / 2 * BM{0})])), (&(((uint8_t*)sign)[(k_outer * BBK{0} / 3 / 8 * BM{0})])));\n\
    }}\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i++) {{\n\
        ((bitnet_float_type*)C)[i] = (bitnet_float_type)((float)(((int32_t*)CBits)[i]) * ((bitnet_float_type*)LUT_Scales)[0] * ((bitnet_float_type*)Scales)[0]);\n\
    }}\n\
  return 0;\n\
}}\n\
\n\
template<int BATCH_SIZE>\n\
int32_t two_qgemm_lut_{0}(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    alignas(32) uint32_t CBits[BATCH_SIZE * BM{0}];\n\
    memset(&(CBits[0]), 0, BATCH_SIZE * BM{0} * sizeof(int32_t));\n\
#pragma unroll\n\
    for (int32_t k_outer = 0; k_outer < {2} / 32; ++k_outer) {{\n\
        two_tbl_impl_{0}<BATCH_SIZE, {2}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 16)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM{0})])));\n\
    }}\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i++) {{\n\
        ((bitnet_float_type*)C)[i] += (bitnet_float_type)((float)(((int32_t*)CBits)[i]) * ((bitnet_float_type*)LUT_Scales)[0] * ((bitnet_float_type*)Scales)[0]);\n\
    }}\n\
  return 0;\n\
}}\n\
\n\
".format(pre, k_list[1], k_list[0])])
    return kernel_code

def gen_top_api(kernel_shapes, k_list):

    kernel_code = "void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* Three_LUT_Scales, void* Two_LUT_Scales, void* Three_QLUT, void* Two_QLUT) {{\n\
    partial_max_reset(bs, (&(((bitnet_float_type*)Three_LUT_Scales)[0])));\n\
    partial_max_reset(bs, (&(((bitnet_float_type*)Two_LUT_Scales)[0])));\n\
    for (int32_t b = 0; b < bs; b++) {{\n\
        for (int32_t k_outer = 0; k_outer < (three_k + two_k) / 24; ++k_outer) {{\n\
            three_partial_max((&(((bitnet_float_type*)Three_LUT_Scales)[b])), (&(((bitnet_float_type*)B)[(k_outer * 24)])));\n\
        }}\n\
        for (int32_t k_outer = 0; k_outer < (three_k + two_k) / 16; ++k_outer) {{\n\
            two_partial_max((&(((bitnet_float_type*)Two_LUT_Scales)[b])), (&(((bitnet_float_type*)B)[(k_outer * 16)])));\n\
        }}\n\
    }}\n\
    if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 16])), (&(((bitnet_float_type*)B)[b * (three_k + two_k)])), (&(((bitnet_float_type*)Three_LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 16])), (&(((bitnet_float_type*)B)[b * (three_k + two_k) + {2}])), (&(((bitnet_float_type*)Two_LUT_Scales)[b])));\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], k_list[0][0], k_list[0][1])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 16])), (&(((bitnet_float_type*)B)[b * (three_k + two_k)])), (&(((bitnet_float_type*)Three_LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 16])), (&(((bitnet_float_type*)B)[b * (three_k + two_k) + {2}])), (&(((bitnet_float_type*)Two_LUT_Scales)[b])));\n\
        }}\n\
    }}\n".format(kernel_shapes[i][0], k_list[i][0], k_list[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])


    kernel_code = "".join([kernel_code, "void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1], k_list[0][0], k_list[0][1], "{}_{}".format(kernel_shapes[0][0], kernel_shapes[0][1]))])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[i][0], kernel_shapes[i][1], k_list[i][0], k_list[i][1], "{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]))])
    kernel_code = "".join([kernel_code, "}\n"])
    return kernel_code

def gen_transform_code(kernel_shapes):
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {\n\
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    const int lut_scales_size = 1;\n\
    int bk = 0;\n\
    int bm = 0;\n"

    kernel_code = "".join([kernel_code, "\n\
    if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[0][0], kernel_shapes[0][1])])

    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "else if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])

    kernel_code = "".join([kernel_code, "\n\
    const int n_tile_num = m / bm;\n\
    const int BK = bk;\n\
    uint8_t * qweights;\n\
    bitnet_float_type * scales;\n\
\n\
    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));\n\
    qweights = (uint8_t *) tensor->data;\n\
    int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;\n\
    nbytes = 32 - nbytes % 32 + nbytes;\n\
    float * i2_scales = (float * )(qweights + nbytes);\n\
\n"])

    kernel_code = "".join([kernel_code, "\
    scales[0] = (bitnet_float_type) i2_scales[0];\n"])

    kernel_code = "".join([kernel_code, "\n\
    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;\n\
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {\n\
        /* .lut_scales_size = */ lut_scales_size,\n\
        /* .BK              = */ BK,\n\
        /* .n_tile_num      = */ n_tile_num,\n\
        /* .qweights        = */ qweights,\n\
        /* .scales          = */ scales\n\
    };\n\
}\n"])

    return kernel_code

def get_three_k_two_k(K, bk):
    bk_num = K // bk
    three_k = bk_num * bk
    two_k = K - three_k
    return two_k, three_k

if __name__ == "__main__":
    ModelShapeDict = {
        "bitnet_b1_58-large"                : [[1536, 4096],
                                               [1536, 1536],
                                               [4096, 1536]],
        "bitnet_b1_58-3B"                   : [[3200, 8640],
                                               [3200, 3200],
                                               [8640, 3200]],
        "Llama3-8B-1.58-100B-tokens"        : [[14336, 4096],
                                               [4096, 14336],
                                               [1024, 4096],
                                               [4096, 4096]] 
    }

    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--model',default="input", type=str, dest="model", 
                        help="choose from bitnet_b1_58-large/bitnet_b1_58-3B/Llama3-8B-1.58-100B-tokens.")
    parser.add_argument('--BM',default="input", type=str,
                        help="block length when cutting one weight (M, K) into M / BM weights (BM, K).")
    parser.add_argument('--BK',default="input", type=str,
                        help="block length when cutting one weight (M, K) into K / BK weights (M, BK).")
    parser.add_argument('--bm',default="input", type=str,
                        help="using simd instructions to compute (bm, 192 / bm) in one block")
    args = parser.parse_args()

    kernel_shapes = ModelShapeDict[args.model]

    BM_list = [int(item) for item in args.BM.split(',')]
    BK_list = [int(item) for item in args.BK.split(',')]
    bm_list = [int(item) for item in args.bm.split(',')]

    tbl_impl_code = []
    k_list = []

    for i in range(len(kernel_shapes)):
        k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

    for i in range(len(kernel_shapes)):
        tbl_impl_code.append(
            gen_tbl_impl("{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], k_list[i])
        )

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))
    
    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert (kernel_shapes[i][1] % BK_list[i]) % 32 == 0, "K %% BK %% 32 should be 0"
        assert bm_list[i] in [16, 32], "choose bm from [16, 32]"

    ctor_code = gen_ctor_code()
    api_code = gen_top_api(kernel_shapes, k_list)
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join("#if defined(GGML_BITNET_TL2_LOSS)"))
        f.write(''.join(ctor_code))
        for code in tbl_impl_code:
            f.write(''.join(code))
        f.write(''.join(api_code))
        f.write(''.join(trans_code))
        f.write(''.join("#endif"))

    config = ConfigParser()

    for i in range(len(kernel_shapes)):
        config.add_section('Kernels_{}'.format(i))
        config.set('Kernels_{}'.format(i), 'M'.format(i), str(kernel_shapes[i][0]))
        config.set('Kernels_{}'.format(i), 'K'.format(i), str(kernel_shapes[i][1]))
        config.set('Kernels_{}'.format(i), 'BM'.format(i), str(BM_list[i]))
        config.set('Kernels_{}'.format(i), 'BK'.format(i), str(BK_list[i]))
        config.set('Kernels_{}'.format(i), 'bmm'.format(i), str(bm_list[i]))

    with open(''.join([output_dir, "/kernel_config.ini"]), 'w') as configfile:
        config.write(configfile)