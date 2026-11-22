#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr, const float * res_ptr, const float res_scale,
                                    const int64_t src0_nb0, const int64_t src0_nb1, const int64_t src0_nb2, const int64_t src1_nb1,
                                    float * dst_ptr, const int64_t dst_nb0, const int64_t dst_nb1, const int64_t dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int64_t stride_x = src0_nb1 / sizeof(float);
    const int64_t stride_w = src1_nb1 / sizeof(float);
    const int64_t stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        if (res_ptr != nullptr) {
            sumf = (res_ptr[bidx * (dst_nb2 / sizeof(float)) + i * stride_y + bidy * split_d_inner + tid] + sumf) * res_scale;
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias, const float * __restrict__ res, const float res_scale,
                                               const int64_t src0_nb0, const int64_t src0_nb1, const int64_t src0_nb2,
                                               const int64_t src1_nb1, float * __restrict__ dst, const int64_t dst_nb0,
                                               const int64_t dst_nb1, const int64_t dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int64_t stride_x = src0_nb1 / sizeof(float);
    const int64_t stride_w = src1_nb1 / sizeof(float);
    const int64_t stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        if (res != nullptr) {
            sumf = (res[bidx * (dst_nb2 / sizeof(float)) + (bidz * split_n_t + i) * stride_y + bidy * split_d_inner + tid] + sumf) * res_scale;
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const float * res, const float res_scale, const int64_t src0_nb0, const int64_t src0_nb1,
                              const int64_t src0_nb2, const int64_t src1_nb1, float * dst, const int64_t dst_nb0, const int64_t dst_nb1,
                              const int64_t dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, res, res_scale, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, res, res_scale, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 2:  launch_kernel(std::integral_constant<int, 2 >{}); break;
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst,
        ggml_tensor * res_add_node, ggml_tensor * scale_node) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // (residual + conv)*scale: the residual is the non-conv operand of the ADD node, the SCALE node is the output
    const bool fuse_res = res_add_node != nullptr;
    GGML_ASSERT(!fuse_res || (!fuse_bias && !fuse_silu && scale_node != nullptr));
    const struct ggml_tensor * res = fuse_res ? (res_add_node->src[0] == dst ? res_add_node->src[1] : res_add_node->src[0]) : nullptr;
    float res_scale = 1.0f;
    if (fuse_res) {
        memcpy(&res_scale, scale_node->op_params, sizeof(float));
        GGML_ASSERT(res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_are_same_shape(res, dst));
    }

    // When fusing, write to the last fused node (the node downstream references).
    const struct ggml_tensor * out = fuse_res ? scale_node : fuse_silu ? silu_dst : dst;
    GGML_ASSERT(!fuse_res || ggml_is_contiguous(out));

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    const float * res_d  = fuse_res ? (const float *) res->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, res_d, res_scale, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, res_d, res_scale, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// short convolution over [past, cur] with the past (d_conv - 1 columns per channel) read from the recurrent state, a
// residual add and a scale, and the new past written back: CONCAT -> VIEW -> VIEW -> CPY -> SSM_CONV -> ADD -> SCALE
// one thread per channel and sequence, so the state may be read and written in place
template <size_t d_conv>
static __global__ void ssm_conv_hist_f32(
        const float * __restrict__ past, const int64_t past_s1, const int64_t past_s2,
        const float * cur, const int64_t cur_s0, const int64_t cur_s1, const int64_t cur_s2,
        const float * __restrict__ w_ptr, const int64_t w_s1,
        const float * res, const int64_t res_s1, const int64_t res_s2, const float res_scale,
        float * y, const int64_t y_s1, const int64_t y_s2,
        float * state, const int64_t state_s1,
        const int64_t nr, const int64_t n_t) {
    const int64_t c = (int64_t) blockIdx.y*blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.x;
    if (c >= nr) {
        return;
    }

    float x[d_conv];
    float w[d_conv];
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_ptr[c*w_s1 + j];
    }
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        x[j] = past[s*past_s2 + c*past_s1 + j];
    }
    const float b = 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        x[d_conv - 1] = cur[s*cur_s2 + i*cur_s0 + c*cur_s1];

        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[j] * w[j];
        }
        sumf += b;
        sumf = (res[s*res_s2 + i*res_s1 + c] + sumf) * res_scale;
        y[s*y_s2 + i*y_s1 + c] = sumf;

#pragma unroll
        for (size_t j = 0; j < d_conv - 1; j++) {
            x[j] = x[j + 1];
        }
    }

#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        state[s*state_s1 + c*(d_conv - 1) + j] = x[j];
    }
}

bool ggml_cuda_ssm_conv_hist_supported(ggml_tensor * const * n) {
    const ggml_tensor * concat = n[0];
    const ggml_tensor * cpy    = n[3];
    const ggml_tensor * conv   = n[4];
    const ggml_tensor * add    = n[5];
    const ggml_tensor * scale  = n[6];
    const ggml_tensor * past   = concat->src[0];
    const ggml_tensor * cur    = concat->src[1];
    const ggml_tensor * w      = conv->src[1];
    const ggml_tensor * res    = add->src[0] == conv ? add->src[1] : add->src[0];
    const ggml_tensor * last   = cpy->src[0];
    const ggml_tensor * cell   = cpy->src[1];

    const int64_t nc  = w->ne[0];
    const int64_t nr  = conv->ne[0];
    const int64_t n_t = conv->ne[1];
    const int64_t n_s = conv->ne[2];

    if (nc != 4 || n_t > 32 || ggml_get_op_params_i32(concat, 0) != 0) {
        return false;
    }
    for (const ggml_tensor * t : { concat, past, cur, w, res, last, cell, conv, add, scale }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // the inputs: the past [nc - 1, nr, n_s] and cur [n_t, nr, n_s], each with contiguous rows of past or one element
    if (past->ne[0] != nc - 1 || past->ne[1] != nr || past->ne[2] != n_s || past->nb[0] != sizeof(float) ||
        cur->ne[0] != n_t || cur->ne[1] != nr || cur->ne[2] != n_s || concat->ne[0] != nc - 1 + n_t || !ggml_is_contiguous(concat)) {
        return false;
    }
    // the stored past is the last nc - 1 columns of the conv input, copied into nr*(nc - 1) contiguous floats per sequence
    if (last->view_src != concat || last->view_offs != (size_t) n_t*sizeof(float) || last->ne[0] != nc - 1 ||
        last->ne[1] != nr || last->ne[2] != n_s || last->nb[1] != concat->nb[1] || last->nb[2] != concat->nb[2] ||
        cell->ne[0] != (nc - 1)*nr || cell->ne[1] != n_s || cell->nb[0] != sizeof(float) || ggml_nelements(cell) != ggml_nelements(last)) {
        return false;
    }
    if (!ggml_are_same_shape(res, conv) || !ggml_is_contiguous(res) || !ggml_is_contiguous(scale) || w->nb[0] != sizeof(float)) {
        return false;
    }
    float bias;
    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
    if (bias != 0.0f) {
        return false;
    }

    // memory: the output may be the residual itself (element for element); the state may be the past itself (one
    // thread reads its channel before it writes it), otherwise no overlaps between written and read memory
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if ((overlap(scale, res) && scale->data != res->data) || overlap(scale, past) || overlap(scale, w) || overlap(scale, cell)) {
        return false;
    }
    if (overlap(scale, cur) && !(scale->data == cur->data && cur->nb[1] == sizeof(float) && cur->nb[0] == scale->nb[1] &&
                                 cur->nb[2] == scale->nb[2])) {
        return false;
    }
    if ((overlap(cell, past) && !(cell->data == past->data && n_s == 1 && past->nb[1] == (nc - 1)*sizeof(float))) ||
        overlap(cell, cur) || overlap(cell, res) || overlap(cell, w)) {
        return false;
    }
    return true;
}

void ggml_cuda_op_ssm_conv_hist(ggml_backend_cuda_context & ctx, ggml_tensor * const * n) {
    const ggml_tensor * concat = n[0];
    const ggml_tensor * cpy    = n[3];
    const ggml_tensor * conv   = n[4];
    const ggml_tensor * add    = n[5];
    ggml_tensor       * scale  = n[6];
    const ggml_tensor * past   = concat->src[0];
    const ggml_tensor * cur    = concat->src[1];
    const ggml_tensor * w      = conv->src[1];
    const ggml_tensor * res    = add->src[0] == conv ? add->src[1] : add->src[0];
    const ggml_tensor * cell   = cpy->src[1];

    float res_scale;
    memcpy(&res_scale, scale->op_params, sizeof(float));

    const int64_t nr  = conv->ne[0];
    const int64_t n_t = conv->ne[1];
    const int64_t n_s = conv->ne[2];

    const int threads = 128;
    const dim3 blocks(n_s, (nr + threads - 1)/threads, 1);
    const size_t f = sizeof(float);
    ssm_conv_hist_f32<4><<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) past->data, past->nb[1]/f, past->nb[2]/f,
        (const float *) cur->data, cur->nb[0]/f, cur->nb[1]/f, cur->nb[2]/f,
        (const float *) w->data, w->nb[1]/f,
        (const float *) res->data, res->nb[1]/f, res->nb[2]/f, res_scale,
        (float *) scale->data, scale->nb[1]/f, scale->nb[2]/f,
        (float *) cell->data, cell->nb[1]/f,
        nr, n_t);
}
