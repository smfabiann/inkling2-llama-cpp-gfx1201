#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
        ggml_tensor * res_add_node = nullptr, ggml_tensor * scale_node = nullptr);

// CONCAT -> VIEW -> VIEW -> CPY -> SSM_CONV -> ADD -> SCALE: a short convolution with its past in the recurrent state
bool ggml_cuda_ssm_conv_hist_supported(ggml_tensor * const * nodes);
void ggml_cuda_op_ssm_conv_hist(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes);
