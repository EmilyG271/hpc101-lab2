// Main task: optimize the MoE forward pass.
//
// Correctness-repaired version of the student's original AMX/AVX-512 design:
//   router -> Top-K -> per-token W8A8 quantization -> expert grouping
//   -> AMX INT8 GEMM -> SwiGLU -> requantization -> AMX down projection.

#include "moe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#if defined(__AMX_INT8__) && defined(__AMX_TILE__) && defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

// -----------------------------------------------------------------------------
// Routing scratch buffers
// -----------------------------------------------------------------------------

static float g_logits[MAX_NUM_TOKENS][MAX_NUM_EXPERTS];
// Store the ORIGINAL affinity sigmoid(logit). The load-balancing bias is used
// only while selecting Top-K and must not enter the normalized gate weights.
static float g_affinity[MAX_NUM_TOKENS][MAX_NUM_EXPERTS];
static int g_topk_indices[MAX_NUM_TOKENS][MAX_TOP_K];
static float g_topk_weights[MAX_NUM_TOKENS][MAX_TOP_K];
static int g_expert_token_count[MAX_NUM_EXPERTS];
static int g_expert_token_list[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
// 分发阶段同时保存路由权重与原 Top-K 槽位，专家回写时无需再次扫描 K。
static float g_expert_token_weight[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
static uint8_t g_expert_topk_slot[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];

// -----------------------------------------------------------------------------
// Compact runtime-layout activation/output buffers
//
// These are flat on purpose. A declaration such as [MAX_NUM_TOKENS][MAX_D_FF]
// has a physical row stride of MAX_D_FF, which is wrong when the runtime H is
// smaller. AMX must see the actual compact [M, K] / [M, N] row strides.
// -----------------------------------------------------------------------------

alignas(64) static int8_t g_quantized_x[MAX_NUM_TOKENS * MAX_D_MODEL];
static float g_x_scale[MAX_NUM_TOKENS];

alignas(64) static int32_t g_shared_gate_out[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int32_t g_shared_up_out[MAX_NUM_TOKENS * MAX_D_FF];
static float g_shared_gated_fp32[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int8_t
    g_shared_quantized_gated[MAX_NUM_TOKENS * MAX_D_FF];
static float g_shared_gated_scale[MAX_NUM_TOKENS];
alignas(64) static int32_t
    g_shared_down_out[MAX_NUM_TOKENS * MAX_D_MODEL];

alignas(64) static int8_t g_amx_A[MAX_NUM_TOKENS * MAX_D_MODEL];
alignas(64) static int32_t g_gate_out[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int32_t g_up_out[MAX_NUM_TOKENS * MAX_D_FF];
static float g_gated_fp32[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int8_t
    g_quantized_gated[MAX_NUM_TOKENS * MAX_D_FF];
static float g_gated_scale[MAX_NUM_TOKENS];
alignas(64) static int32_t g_down_out[MAX_NUM_TOKENS * MAX_D_MODEL];

// -----------------------------------------------------------------------------
// Prepacked expert weights
//
// Logical multiplication is A[M,K] * B[K,N]. The framework stores each weight
// matrix output-major as W[N,K], so preprocess transposes and VNNI-packs it as
// B_pack[K/4, N*4]:
//   B_pack[k/4][4*n + k%4] = W[n][k].
// -----------------------------------------------------------------------------

alignas(64) static int8_t
    g_packed_gate[MAX_NUM_EXPERTS][MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_up[MAX_NUM_EXPERTS][MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_down[MAX_NUM_EXPERTS][MAX_D_FF * MAX_D_MODEL];

static float g_scale_gate[MAX_NUM_EXPERTS];
static float g_scale_up[MAX_NUM_EXPERTS];
static float g_scale_down[MAX_NUM_EXPERTS];

alignas(64) static int8_t
    g_packed_shared_gate[MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_shared_up[MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_shared_down[MAX_D_FF * MAX_D_MODEL];
static float g_scale_shared_gate;
static float g_scale_shared_up;
static float g_scale_shared_down;

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
alignas(64) static thread_local unsigned char g_tile_cfg[64];
// Tile 配置属于线程的扩展寄存器状态；缓存当前行数可跳过重复 LDTILECFG。
static thread_local int g_last_tile_rows = -1;
#endif
static bool g_amx_runtime_enabled = false;

/**
 * @brief 将框架的输出通道优先 INT8 权重转置并打包为 AMX TDPBSSD 所需布局。
 *
 * 功能：框架中的权重按 W[N][K] 连续存储，而矩阵乘逻辑需要 B[K][N]。
 * 本函数在转置的同时，将 K 维每连续 4 个有符号 INT8 元素组成一个 4-byte
 * 点积小组，最终得到 B_pack[K/4][N*4]。打包后的地址映射为：
 *     B_pack[k/4][4*n + k%4] = W[n][k]。
 *
 * 实现思路：外层遍历 K/4 个四元素条带，中层遍历输出列 n，内层复制该列
 * 在 K 维上的 4 个连续元素。该操作只在 preprocess() 中执行一次，前向阶段
 * 可以直接顺序加载打包权重，避免每次矩阵乘时临时转置或跨行收集。
 *
 * 向量化情况：本函数本身未显式使用 AVX-512/AMX；它属于 AMX 计算前的数据
 * 预处理。真正的矩阵计算由 amx_matmul_packed() 使用 AMX TDPBSSD 完成。
 *
 * @param src    原始权重，逻辑形状为 [N, K]，元素类型为 int8_t。
 * @param packed 输出缓冲区，逻辑形状为 [K/4, N*4]。
 * @param K      归约维度，实验保证为 64 的倍数。
 * @param N      输出维度，实验保证为 64 的倍数。
 */
static void pack_output_major_weight(const int8_t* src, int8_t* packed,
                                     int K, int N) {
    // src is [N][K], while logical B is [K][N]. K is guaranteed to be a
    // multiple of 64 by the driver, hence also a multiple of 4.
    for (int k4 = 0; k4 < K / 4; ++k4) {
        for (int n = 0; n < N; ++n) {
            for (int r = 0; r < 4; ++r) {
                packed[(size_t)k4 * (N * 4) + 4 * n + r] =
                    src[(size_t)n * K + 4 * k4 + r];
            }
        }
    }
}

/**
 * @brief 为当前进程/线程申请使用 Intel AMX Tile 数据状态的权限。
 *
 * 功能：Linux 内核采用扩展状态按需授权机制。程序在第一次执行 AMX Tile 指令
 * 之前，需要通过 arch_prctl 请求 XFEATURE_XTILEDATA 权限，否则可能在执行
 * LDTILECFG、TILELOADD 或 TDPBSSD 时触发异常。
 *
 * 实现思路：仅当编译器同时定义 __AMX_INT8__ 和 __AMX_TILE__ 时启用 AMX
 * 代码。Linux 下调用 syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
 * XFEATURE_XTILEDATA)；其他 AMX 编译平台直接返回 true；未启用 AMX 的构建
 * 返回 false，使后续矩阵乘自动使用标量回退实现。
 *
 * 向量化情况：不执行数值向量化；该函数负责 AMX 运行环境初始化和能力选择。
 *
 * @return true 表示可以进入 AMX kernel；false 表示使用标量回退路径。
 */
static bool request_amx_permission() {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
#if defined(__linux__)
    // Linux requires each process/thread to request permission for XTILEDATA
    // before executing the first AMX instruction.
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                   XFEATURE_XTILEDATA) == 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

/**
 * @brief 在计时开始前预打包全部路由专家和共享专家的 INT8 权重。
 *
 * 功能：读取 MoEWeights 中已经完成 W8 量化的 Gate、Up、Down 权重，将它们
 * 转换为 AMX TDPBSSD 可直接消费的 VNNI 风格布局，并保存框架提供的每矩阵
 * FP32 反量化 scale。原始权重不会被覆盖，满足实验对 preprocess() 的要求。
 *
 * 实现思路：
 * 1. Gate/Up 的原始形状是 [H,D]，按 K=D、N=H 打包；
 * 2. Down 的原始形状是 [D,H]，按 K=H、N=D 打包；
 * 3. 每个专家分别保存 s_gate、s_up、s_down；
 * 4. 共享专家执行同样处理；
 * 5. 最后申请 AMX 权限，决定前向阶段使用 AMX 还是标量回退 kernel。
 *
 * 向量化情况：本函数不直接执行向量矩阵运算；其数据重排专门服务于后续 AMX
 * 分块矩阵乘。预打包只执行一次，避免在每轮 moe_forward_optimized() 中重复。
 *
 * @param w 模型权重及运行时维度。函数只读权重内容，不原地修改权重数组。
 */
void preprocess(MoEWeights& w) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const size_t expert_elems = (size_t)D * H;

    // The framework has already quantized all expert weights to int8 and
    // provides their dequantization scales. preprocess only repacks them.
    for (int e = 0; e < E; ++e) {
        pack_output_major_weight(w.w_gate + (size_t)e * expert_elems,
                                 g_packed_gate[e], D, H);
        pack_output_major_weight(w.w_up + (size_t)e * expert_elems,
                                 g_packed_up[e], D, H);
        pack_output_major_weight(w.w_down + (size_t)e * expert_elems,
                                 g_packed_down[e], H, D);
        g_scale_gate[e] = w.s_gate[e];
        g_scale_up[e] = w.s_up[e];
        g_scale_down[e] = w.s_down[e];
    }

    pack_output_major_weight(w.sh_gate, g_packed_shared_gate, D, H);
    pack_output_major_weight(w.sh_up, g_packed_shared_up, D, H);
    pack_output_major_weight(w.sh_down, g_packed_shared_down, H, D);
    g_scale_shared_gate = w.sh_s_gate;
    g_scale_shared_up = w.sh_s_up;
    g_scale_shared_down = w.sh_s_down;

    g_amx_runtime_enabled = request_amx_permission();
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    g_last_tile_rows = -1;
#endif
}

// -----------------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------------

/**
 * @brief 计算每个 token 对每个路由专家的 FP32 Router logit。
 *
 * 功能：执行 z[t,e] = dot(x[t,:], w_router[e,:])，结果写入 g_logits。
 * Router 按实验要求保持 FP32，不参与 W8A8 专家矩阵的整数计算。
 *
 * 实现思路：外层遍历 token，AVX-512 路径每次并行处理 4 个专家，复用同一
 * 个 x 向量并维护 4 条独立 FMA 累加链；每条指令处理 16 个 FP32，最后通过
 * _mm512_reduce_add_ps 做水平归约。x 和 w_router 由普通 new[] 分配，
 * 不保证 64 字节对齐，因此必须使用 _mm512_loadu_ps，避免对齐加载异常。
 *
 * 向量化情况：
 * - 定义 __AVX512F__ 时：使用 AVX-512 FP32 FMA，每条向量处理 16 个元素；
 * - 未定义时：使用标量点积，保证普通编译器和非 AVX-512 平台可运行。
 *
 * @param x          输入 token，紧凑布局 [num_tokens,D]。
 * @param w          提供 FP32 Router 权重 w_router。
 * @param num_tokens token 数量。
 * @param D          模型隐藏维度。
 * @param E          路由专家数量。
 */
static inline void compute_logits(const float* x, const MoEWeights& w,
                                  int num_tokens, int D, int E) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* x_t = x + (size_t)t * D;
#if defined(__AVX512F__)
        int e = 0;
        // 一次计算 4 个专家：同一个 x_vec 被复用 4 次，减少输入向量重复加载，
        // 同时形成 4 条互不依赖的 FMA 累加链，提高指令级并行度。
        for (; e + 3 < E; e += 4) {
            const float* r0 = w.w_router + (size_t)(e + 0) * D;
            const float* r1 = w.w_router + (size_t)(e + 1) * D;
            const float* r2 = w.w_router + (size_t)(e + 2) * D;
            const float* r3 = w.w_router + (size_t)(e + 3) * D;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                acc0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r0 + d), acc0);
                acc1 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r1 + d), acc1);
                acc2 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r2 + d), acc2);
                acc3 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r3 + d), acc3);
            }
            g_logits[t][e + 0] = _mm512_reduce_add_ps(acc0);
            g_logits[t][e + 1] = _mm512_reduce_add_ps(acc1);
            g_logits[t][e + 2] = _mm512_reduce_add_ps(acc2);
            g_logits[t][e + 3] = _mm512_reduce_add_ps(acc3);
        }

        // E 不一定是 4 的倍数，处理最后 0~3 个专家。
        for (; e < E; ++e) {
            const float* r_e = w.w_router + (size_t)e * D;
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                acc = _mm512_fmadd_ps(
                    xv, _mm512_loadu_ps(r_e + d), acc);
            }
            g_logits[t][e] = _mm512_reduce_add_ps(acc);
        }
#else
        for (int e = 0; e < E; ++e) {
            const float* r_e = w.w_router + (size_t)e * D;
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) sum += r_e[d] * x_t[d];
            g_logits[t][e] = sum;
        }
#endif
    }
}

/**
 * @brief 计算 Sigmoid 亲和度，并按“亲和度 + 负载均衡偏置”选择 Top-K 专家。
 *
 * 功能：先计算 s[t,e] = sigmoid(g_logits[t,e])，再依据 s[t,e] + bias[e]
 * 选择 K 个专家。偏置只影响专家选择，绝不能进入后续路由权重归一化。
 *
 * 实现思路：K=2 时一次扫描同时维护前两名；其他 K 使用 used[] 逐轮选择。
 * 两条路径均按专家编号递增扫描且只在严格大于时替换，因此同分时小索引优先，
 * 与标量参考实现的 tie-break 规则一致。
 *
 * 向量化情况：未显式向量化。Sigmoid 依赖 expf，Top-K 又包含数据相关分支，
 * 当前保持标量实现以优先保证选择语义与参考版本一致。
 *
 * @param w          提供每个专家的负载均衡 bias。
 * @param num_tokens token 数量。
 * @param E          专家数量。
 * @param K          每个 token 选择的专家数量。
 */
static inline void compute_affinity_and_topk(const MoEWeights& w,
                                              int num_tokens, int E, int K) {
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < E; ++e) {
            const float z = g_logits[t][e];
            g_affinity[t][e] = 1.0f / (1.0f + expf(-z));
        }

        // K=2 是 S4 的固定热点：一次扫描同时维护第一、第二名。
        // 按专家编号递增扫描且只在严格大于时替换，因此同分时仍是小索引优先。
        if (K == 2) {
            int best0 = -1;
            int best1 = -1;
            float score0 = 0.0f;
            float score1 = 0.0f;
            for (int e = 0; e < E; ++e) {
                const float score = g_affinity[t][e] + w.bias[e];
                if (best0 < 0 || score > score0) {
                    best1 = best0;
                    score1 = score0;
                    best0 = e;
                    score0 = score;
                } else if (best1 < 0 || score > score1) {
                    best1 = e;
                    score1 = score;
                }
            }
            g_topk_indices[t][0] = best0;
            g_topk_indices[t][1] = best1;
        } else {
            // 通用 K 路径与参考实现一致：逐轮选择，严格大于保证小索引优先。
            bool used[MAX_NUM_EXPERTS] = {};
            for (int k = 0; k < K; ++k) {
                int best = -1;
                for (int e = 0; e < E; ++e) {
                    if (used[e]) continue;
                    if (best < 0 ||
                        g_affinity[t][e] + w.bias[e] >
                            g_affinity[t][best] + w.bias[best]) {
                        best = e;
                    }
                }
                used[best] = true;
                g_topk_indices[t][k] = best;
            }
        }
    }
}

/**
 * @brief 对选中专家的原始 Sigmoid 亲和度进行归一化，得到路由权重。
 *
 * 功能：对每个 token 计算
 *     gate[t,k] = affinity[t, topk[k]] / sum_selected_affinity。
 * 注意这里既不是 exp(logit) Softmax，也不能把负载均衡 bias 加入分母或分子。
 *
 * 实现思路：第一遍对 K 个选中专家求和，第二遍逐个除以总和并写入
 * g_topk_weights。由于实验 MAX_TOP_K=4，循环规模很小。
 *
 * 向量化情况：未显式向量化。K 最大仅为 4，使用 SIMD 的装载、掩码和归约
 * 开销通常高于收益，且这里不是主要计算热点。
 *
 * @param num_tokens token 数量。
 * @param K          Top-K 数量。
 */
static inline void compute_topk_weights(int num_tokens, int K) {
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += g_affinity[t][g_topk_indices[t][k]];
        }
        for (int k = 0; k < K; ++k) {
            g_topk_weights[t][k] =
                g_affinity[t][g_topk_indices[t][k]] / sum;
        }
    }
}

/**
 * @brief 将 token 按其 Top-K 路由结果聚拢到各个专家的连续 token 列表中。
 *
 * 功能：构建专家 token 列表，并同步保存每个分发项的路由权重和原 Top-K
 * 槽位。这既把逐 token 调用转换为逐专家 GEMM，也消除了专家回写时的 K 次搜索。
 *
 * 实现思路：先清零 E 个专家的计数器，再遍历所有 token 的 K 个专家编号，
 * 将 token ID 追加到对应专家列表。随后每个专家可一次 Gather 出紧凑 A 矩阵，
 * 使 AMX 在 M 维同时处理多个 token，提高权重复用率。
 *
 * 向量化情况：未使用 AVX-512/AMX。本阶段主要是间接索引、计数和离散写入，
 * 不适合直接向量化；它为后续 AMX GEMM 创造连续矩阵布局。
 *
 * @param num_tokens token 数量。
 * @param E          专家数量。
 * @param K          每个 token 选择的专家数量。
 */
static inline void dispatch_tokens_to_experts(int num_tokens, int E, int K) {
    std::memset(g_expert_token_count, 0, (size_t)E * sizeof(int));
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = g_topk_indices[t][k];
            const int slot = g_expert_token_count[e]++;
            g_expert_token_list[e][slot] = t;
            g_expert_token_weight[e][slot] = g_topk_weights[t][k];
            g_expert_topk_slot[e][slot] = (uint8_t)k;
        }
    }
}

// -----------------------------------------------------------------------------
// Per-token activation quantization
// -----------------------------------------------------------------------------

/**
 * @brief 将 FP32 输入按 token 独立量化为 INT8，并记录 per-token scale。
 *
 * 功能：对每个 token 计算 max_abs，得到 s_x=max_abs/127；全零 token 使用
 * scale=1。随后执行 x_q=round(x/s_x)，并将结果限制到 [-127,127]。
 * 输出写入紧凑布局 g_quantized_x[t*D+d]，scale 写入 g_x_scale[t]。
 *
 * 实现思路：每个 token 分两遍处理。第一遍求绝对值最大值，第二遍执行除法、
 * 最近偶数舍入、饱和和窄化。使用除法而非倒数近似乘法，以尽量贴近参考实现
 * 在重量化边界附近的数值行为。
 *
 * 向量化情况：
 * - AVX-512 路径：ZMM 每次处理 16 个 FP32；使用向量绝对值、最大值归约、
 *   除法、roundscale、clamp 和 FP32->INT32 转换；最后写成 16 个 INT8；
 * - 标量路径：使用 fabsf/lrintf 完成同一量化语义。
 *
 * @param x          输入 FP32 token，布局 [num_tokens,D]。
 * @param num_tokens token 数量。
 * @param D          每个 token 的元素数量。
 */
static inline void quantize_input(const float* x, int num_tokens, int D) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* x_t = x + (size_t)t * D;
        int8_t* q_t = g_quantized_x + (size_t)t * D;

#if defined(__AVX512F__)
        __m512 max_vec = _mm512_setzero_ps();
        for (int d = 0; d < D; d += 16) {
            const __m512 v = _mm512_loadu_ps(x_t + d);
            max_vec = _mm512_max_ps(max_vec, _mm512_abs_ps(v));
        }
        const float max_abs = _mm512_reduce_max_ps(max_vec);
#else
        float max_abs = 0.0f;
        for (int d = 0; d < D; ++d) {
            max_abs = std::max(max_abs, fabsf(x_t[d]));
        }
#endif

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        g_x_scale[t] = scale;

#if defined(__AVX512F__)
        const __m512 scale_vec = _mm512_set1_ps(scale);
        const __m512 lo = _mm512_set1_ps(-127.0f);
        const __m512 hi = _mm512_set1_ps(127.0f);
        for (int d = 0; d < D; d += 16) {
            __m512 q = _mm512_div_ps(_mm512_loadu_ps(x_t + d), scale_vec);
            q = _mm512_roundscale_ps(
                q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q = _mm512_min_ps(hi, _mm512_max_ps(lo, q));
            const __m512i q_i32 = _mm512_cvtps_epi32(q);
            // 16×INT32 直接窄化为 16×INT8，避免临时数组和 16 次标量写回。
            const __m128i q_i8 = _mm512_cvtepi32_epi8(q_i32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_t + d), q_i8);
        }
#else
        for (int d = 0; d < D; ++d) {
            long q = lrintf(x_t[d] / scale);
            q = std::max(-127L, std::min(127L, q));
            q_t[d] = (int8_t)q;
        }
#endif
    }
}

/**
 * @brief 将一行 SwiGLU FP32 隐藏激活重新量化为 INT8。
 *
 * 功能：计算 s_h=max(abs(h))/127，并执行 h_q=round(h/s_h)。该步骤是 W8A8
 * 流水线中 Gate/Up 的 FP32 SwiGLU 与 Down INT8 矩阵乘之间的重量化边界。
 *
 * 实现思路：SwiGLU 生成 hidden 时已经同步统计 max_abs，本函数直接计算 scale
 * 并完成除法、最近偶数舍入、饱和及 INT32→INT8 窄化，避免再次扫描求最大值。
 * 返回的 s_h 用于 Down 输出反量化：output_fp32=output_int32*s_h*s_down。
 *
 * 向量化情况：AVX-512 路径每次处理 16 个 FP32，并用 VPMOVDB 风格窄化一次
 * 写出 16 个 INT8；非 AVX-512 构建保留 lrintf 标量回退。
 *
 * @param src     一行 FP32 隐藏激活。
 * @param dst     输出 INT8 隐藏激活。
 * @param length  隐藏维度 H。
 * @param max_abs SwiGLU 计算阶段得到的该行绝对值最大值。
 * @return        当前行的激活 scale s_h。
 */
static inline float quantize_hidden_row(const float* src, int8_t* dst,
                                        int length, float max_abs) {
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
#if defined(__AVX512F__)
    const __m512 scale_vec = _mm512_set1_ps(scale);
    const __m512 lo = _mm512_set1_ps(-127.0f);
    const __m512 hi = _mm512_set1_ps(127.0f);
    for (int i = 0; i < length; i += 16) {
        __m512 q = _mm512_div_ps(_mm512_loadu_ps(src + i), scale_vec);
        q = _mm512_roundscale_ps(
            q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm512_min_ps(hi, _mm512_max_ps(lo, q));
        const __m512i q_i32 = _mm512_cvtps_epi32(q);
        const __m128i q_i8 = _mm512_cvtepi32_epi8(q_i32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), q_i8);
    }
#else
    for (int i = 0; i < length; ++i) {
        long q = lrintf(src[i] / scale);
        q = std::max(-127L, std::min(127L, q));
        dst[i] = (int8_t)q;
    }
#endif
    return scale;
}

// -----------------------------------------------------------------------------
// Small-M kernels
//
// 当 M<=4 时，AMX 的 16 行 Tile 严重欠填充。这里直接读取框架原始的输出通道
// 优先权重 W[N,K]，用 AVX-512BW 将 signed INT8 扩展到 INT16，再用 VPMADDWD
// 对相邻两项求和并累加到 INT32。一次同时处理 4 个输出通道以复用 A。
// -----------------------------------------------------------------------------

static constexpr int SMALL_M_THRESHOLD = 4;
// 仅在每行计算量足够大时启用 AVX-512BW 小 M kernel；避免 S1/S4 中
// 矩阵较小时，符号扩展与水平归约开销超过欠填充 AMX 的成本。
static constexpr size_t SMALL_M_WORK_THRESHOLD = 262144;
// 当前实测融合 Gate/Up AMX 在 S3/S4 略慢，保留实现但暂时关闭。
static constexpr bool USE_FUSED_GATE_UP_AMX = false;

static inline bool should_use_small_m_kernel(int M, int K, int N) {
    return M <= SMALL_M_THRESHOLD &&
           (size_t)K * (size_t)N >= SMALL_M_WORK_THRESHOLD;
}

static void small_m_gate_up_output_major(
    const int8_t* A, const int8_t* W_gate, const int8_t* W_up,
    int32_t* C_gate, int32_t* C_up, int M, int K, int N) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 4) {
            __m512i gate_acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};
            __m512i up_acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};

            for (int k0 = 0; k0 < K; k0 += 32) {
                const __m256i a_i8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + k0));
                const __m512i a_i16 = _mm512_cvtepi8_epi16(a_i8);

                for (int j = 0; j < 4; ++j) {
                    const int8_t* wg = W_gate + (size_t)(n0 + j) * K + k0;
                    const int8_t* wu = W_up + (size_t)(n0 + j) * K + k0;
                    const __m512i wg_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(wg)));
                    const __m512i wu_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(wu)));
                    gate_acc[j] = _mm512_add_epi32(
                        gate_acc[j], _mm512_madd_epi16(a_i16, wg_i16));
                    up_acc[j] = _mm512_add_epi32(
                        up_acc[j], _mm512_madd_epi16(a_i16, wu_i16));
                }
            }

            for (int j = 0; j < 4; ++j) {
                C_gate[(size_t)m * N + n0 + j] =
                    _mm512_reduce_add_epi32(gate_acc[j]);
                C_up[(size_t)m * N + n0 + j] =
                    _mm512_reduce_add_epi32(up_acc[j]);
            }
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t gate_acc = 0;
            int32_t up_acc = 0;
            for (int k = 0; k < K; ++k) {
                const int32_t a = (int32_t)A[(size_t)m * K + k];
                gate_acc += a * (int32_t)W_gate[(size_t)n * K + k];
                up_acc += a * (int32_t)W_up[(size_t)n * K + k];
            }
            C_gate[(size_t)m * N + n] = gate_acc;
            C_up[(size_t)m * N + n] = up_acc;
        }
    }
#endif
}

static void small_m_matmul_output_major(
    const int8_t* A, const int8_t* W, int32_t* C,
    int M, int K, int N) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 4) {
            __m512i acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};

            for (int k0 = 0; k0 < K; k0 += 32) {
                const __m256i a_i8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + k0));
                const __m512i a_i16 = _mm512_cvtepi8_epi16(a_i8);
                for (int j = 0; j < 4; ++j) {
                    const int8_t* w_row = W + (size_t)(n0 + j) * K + k0;
                    const __m512i w_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(w_row)));
                    acc[j] = _mm512_add_epi32(
                        acc[j], _mm512_madd_epi16(a_i16, w_i16));
                }
            }

            for (int j = 0; j < 4; ++j) {
                C[(size_t)m * N + n0 + j] =
                    _mm512_reduce_add_epi32(acc[j]);
            }
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)A[(size_t)m * K + k] *
                       (int32_t)W[(size_t)n * K + k];
            }
            C[(size_t)m * N + n] = acc;
        }
    }
#endif
}

// -----------------------------------------------------------------------------
// AMX INT8 GEMM. The scalar branch is a build/runtime fallback; the algorithm
// and packed weight layout are identical, which also makes local verification
// possible on machines or compilers without AMX.
// -----------------------------------------------------------------------------

/**
 * @brief 使用标量指令计算与 AMX 打包格式完全一致的 INT8 GEMM。
 *
 * 功能：计算 C[M,N] = A[M,K] * B[K,N]，INT8 乘积在 INT32 中累加。
 * B 不以普通二维矩阵保存，而是读取 preprocess() 生成的 B_pack 布局。
 *
 * 实现思路：三重循环遍历 m、n、k，通过
 *     B_pack[(k/4)*(N*4) + 4*n + (k&3)]
 * 还原逻辑 B[k,n]。每个 C 元素使用独立 int32_t 累加器，K 最大为 1024，
 * 最坏情况下仍处于 INT32 安全范围内。
 *
 * 向量化情况：没有手写 SIMD/AMX。它是编译器未启用 AMX或运行时 AMX权限
 * 不可用时的正确性回退路径，也用于在普通开发机上验证打包布局。
 *
 * @param A      紧凑布局的左矩阵 [M,K]。
 * @param B_pack AMX/VNNI 打包后的右矩阵 [K/4,N*4]。
 * @param C      INT32 输出矩阵 [M,N]。
 */
static void scalar_matmul_packed(const int8_t* A, const int8_t* B_pack,
                                 int32_t* C, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const int8_t b =
                    B_pack[(size_t)(k / 4) * (N * 4) + 4 * n + (k & 3)];
                acc += (int32_t)A[(size_t)m * K + k] * (int32_t)b;
            }
            C[(size_t)m * N + n] = acc;
        }
    }
}

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
/**
 * @brief 在 64 字节 TILECFG 内存结构中设置一个 Tile 的行数和每行字节数。
 *
 * 功能：写入 tileN.colsb 与 tileN.rows 字段。colsb 使用两个字节的小端格式，
 * rows 使用一个字节。调用者负责保证 rows<=16、bytes_per_row<=64。
 *
 * 实现思路：按照 Intel TILECFG 的固定字节偏移写入配置，不执行 Tile 指令；
 * 最终由 configure_amx_tiles() 调用 _tile_loadconfig 一次性加载到硬件。
 *
 * 向量化情况：不涉及数值向量化，属于 AMX 控制状态配置辅助函数。
 */
static inline void set_tile_shape(int tile, int rows, int bytes_per_row) {
    g_tile_cfg[16 + 2 * tile] = (unsigned char)(bytes_per_row & 0xff);
    g_tile_cfg[17 + 2 * tile] =
        (unsigned char)((bytes_per_row >> 8) & 0xff);
    g_tile_cfg[48 + tile] = (unsigned char)rows;
}

/**
 * @brief 配置融合 Gate/Up 与普通 Down 共用的 AMX Tile 形状。
 *
 * 功能：Tile0/1 分别作为 Gate/Up 的 INT32 累加器，Tile2 保存 A，Tile3/4
 * 保存 Gate/Up 的打包权重。普通 Down 复用 Tile0、Tile2、Tile3。
 * rows_m 用于支持 token 数不是 16 倍数时的 M 维尾块。
 *
 * 实现思路：先将 TILECFG 全部清零，设置 palette_id=1，再调用
 * set_tile_shape() 填写三个 Tile，最后执行 _tile_loadconfig。若当前线程已经
 * 使用相同 rows_m，则直接复用配置。每个 Tile 均满足最多 16 行、每行 64 字节限制。
 *
 * 向量化情况：使用 AMX Tile 配置指令，但不执行乘加；实际矩阵运算由
 * amx_matmul_packed() 中的 _tile_dpbssd 完成。
 *
 * @param rows_m 当前 M 分块的有效 token 行数，范围为 1..16。
 */
static inline void configure_amx_tiles(int rows_m) {
    // Gate/Up/Down 的 Tile K/N 分块均固定为 64/16；只有 M 尾块行数变化。
    // 如果当前线程已经加载相同行数的配置，直接复用硬件 Tile 状态。
    if (g_last_tile_rows == rows_m) return;

    // 一份公共配置同时服务融合 Gate/Up 和普通 Down：
    //   tile0: Gate/普通 C，[rows_m,16] int32
    //   tile1: Up C，[rows_m,16] int32
    //   tile2: A，[rows_m,64] int8
    //   tile3: Gate/普通 B，[16,16*4] int8
    //   tile4: Up B，[16,16*4] int8
    std::memset(g_tile_cfg, 0, sizeof(g_tile_cfg));
    g_tile_cfg[0] = 1;
    set_tile_shape(0, rows_m, 64);
    set_tile_shape(1, rows_m, 64);
    set_tile_shape(2, rows_m, 64);
    set_tile_shape(3, 16, 64);
    set_tile_shape(4, 16, 64);
    _tile_loadconfig(g_tile_cfg);
    g_last_tile_rows = rows_m;
}

/**
 * @brief 使用 Intel AMX TDPBSSD 执行有符号 INT8 GEMM，并在 INT32 中累加。
 *
 * 功能：计算 C[M,N] = A[M,K] * B[K,N]。A 和权重均为 signed INT8，因而
 * 使用 _tile_dpbssd，而不是无符号/有符号混合版本。K 维多个 64 元素分块
 * 在 Tile 0 中持续累加，完成后一次性写回对应的 C 子块。
 *
 * 实现思路：
 * 1. M 维按 16 行分块，尾块通过 rows_m 动态配置；
 * 2. N 维按 16 个 INT32 输出列分块；
 * 3. 每个输出块先 _tile_zero(0)；
 * 4. K 维每次加载 64 个 INT8，B 对应加载 16 行×64 字节的四元素打包块；
 * 5. _tile_dpbssd(0,1,2) 完成 4-way INT8 点积并累加；
 * 6. _tile_stored 将 [rows_m,16] INT32 结果写回紧凑 C 矩阵。
 *
 * 向量化情况：显式使用 Intel AMX-INT8，是本实现主要的矩阵计算加速路径。
 * 实验保证 K、N 均为 64 的倍数，因此不需要 K/N 掩码尾处理。
 *
 * @param A      左矩阵 [M,K]，紧凑 INT8 布局。
 * @param B_pack 右矩阵的 AMX 四元素打包布局 [K/4,N*4]。
 * @param C      输出矩阵 [M,N]，元素为 int32_t。
 */
static void amx_matmul_packed(const int8_t* A, const int8_t* B_pack,
                              int32_t* C, int M, int K, int N) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    // The framework guarantees K and N are multiples of 64.
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        const int rows_m = std::min(TILE_M, M - m0);
        configure_amx_tiles(rows_m);

        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            _tile_zero(0);
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                _tile_loadd(2, A + (size_t)m0 * K + k0, K);
                _tile_loadd(3,
                            B_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                            N * 4);
                _tile_dpbssd(0, 2, 3);
            }
            _tile_stored(0, C + (size_t)m0 * N + n0, N * 4);
        }
    }
}

// Gate 和 Up 具有相同的 A/M/K/N。融合后每个 K-block 只加载一次 A，并用
// tile0/tile1 维护两条独立累加链，分别消费 Gate/Up 的打包权重。
static void amx_matmul_gate_up_packed(
    const int8_t* A, const int8_t* B_gate_pack, const int8_t* B_up_pack,
    int32_t* C_gate, int32_t* C_up, int M, int K, int N) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        const int rows_m = std::min(TILE_M, M - m0);
        configure_amx_tiles(rows_m);

        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            _tile_zero(0);
            _tile_zero(1);
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                _tile_loadd(2, A + (size_t)m0 * K + k0, K);
                _tile_loadd(
                    3,
                    B_gate_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                    N * 4);
                _tile_loadd(
                    4,
                    B_up_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                    N * 4);
                _tile_dpbssd(0, 2, 3);
                _tile_dpbssd(1, 2, 4);
            }
            _tile_stored(0, C_gate + (size_t)m0 * N + n0, N * 4);
            _tile_stored(1, C_up + (size_t)m0 * N + n0, N * 4);
        }
    }
}
#endif

/**
 * @brief 在 AMX kernel 与标量回退 kernel 之间进行统一分派。
 *
 * 功能：向上层隐藏平台差异，使共享专家和路由专家始终调用同一矩阵乘接口。
 * 若编译器启用了 AMX 且运行时权限申请成功，则进入 amx_matmul_packed()；
 * 否则进入 scalar_matmul_packed()，两条路径使用完全相同的 B_pack 布局。
 *
 * 实现思路：通过编译期宏剔除不受支持的 intrinsic，再通过
 * g_amx_runtime_enabled 做运行时选择，从而兼顾目标 Sapphire Rapids 性能和
 * 普通开发环境的可编译、可验证性。
 *
 * 向量化情况：本函数自身不计算；其 AMX 分支显式向量/矩阵化，回退分支为标量。
 */
static void matmul_packed(const int8_t* A, const int8_t* B_pack, int32_t* C,
                          int M, int K, int N) {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (g_amx_runtime_enabled) {
        amx_matmul_packed(A, B_pack, C, M, K, N);
        return;
    }
#endif
    scalar_matmul_packed(A, B_pack, C, M, K, N);
}

// Gate/Up 统一分派：仅当 M<=4 且 K×N 足够大时使用 AVX-512BW 小 M
// kernel；其余情况使用实测更快的分离 Gate/Up AMX。融合实现保留供后续 A/B。
static void matmul_gate_up(
    const int8_t* A,
    const int8_t* B_gate_pack, const int8_t* B_up_pack,
    const int8_t* W_gate, const int8_t* W_up,
    int32_t* C_gate, int32_t* C_up,
    int M, int K, int N) {
    if (should_use_small_m_kernel(M, K, N)) {
        small_m_gate_up_output_major(
            A, W_gate, W_up, C_gate, C_up, M, K, N);
        return;
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (USE_FUSED_GATE_UP_AMX && g_amx_runtime_enabled) {
        amx_matmul_gate_up_packed(
            A, B_gate_pack, B_up_pack, C_gate, C_up, M, K, N);
        return;
    }
#endif
    // 实测分离 Gate/Up 在 S3/S4 更快；matmul_packed 会自行选择 AMX/标量。
    matmul_packed(A, B_gate_pack, C_gate, M, K, N);
    matmul_packed(A, B_up_pack, C_up, M, K, N);
}

// Down/普通单矩阵分派：M<=4 且 K×N 达到阈值时读取原始 W[N,K]；
// 否则保持打包 AMX 路径。
static void matmul_small_m_or_packed(
    const int8_t* A, const int8_t* B_pack, const int8_t* W,
    int32_t* C, int M, int K, int N) {
    if (should_use_small_m_kernel(M, K, N)) {
        small_m_matmul_output_major(A, W, C, M, K, N);
        return;
    }
    matmul_packed(A, B_pack, C, M, K, N);
}

// -----------------------------------------------------------------------------
// Shared and routed experts
// -----------------------------------------------------------------------------

/**
 * @brief 计算所有 token 都必须经过的共享 SwiGLU 专家，并累加到残差输出。
 *
 * 功能：依次完成共享专家的 Gate 投影、Up 投影、FP32 SwiGLU、隐藏激活重新
 * 量化、Down 投影和反量化累加。共享专家路由权重固定为 1。
 *
 * 实现思路：
 * 1. M<=4 且 K×N 足够大时使用 AVX-512BW；否则分离执行 Gate/Up AMX；
 * 2. 使用输入 scale 与权重 scale 将 INT32 投影反量化为 FP32；
 * 3. 计算 SiLU(gate)*up，并逐 token 重新量化为 INT8；
 * 4. 批量执行 INT8 Down GEMM；
 * 5. 乘以 s_h*s_down，直接累加到已经包含残差 x 的 y。
 * 所有矩阵缓冲区均使用运行时 D/H 的紧凑行跨度，避免 MAX_* 二维数组造成
 * 的物理 stride 错误。
 *
 * 向量化情况：常规 Gate/Up/Down 使用 AMX；仅对计算量足够大的小 M 使用
 * AVX-512BW 专用路径。hidden 重量化和 Down 反量化使用 AVX-512。
 */
static void compute_shared_expert(const MoEWeights& w, float* y,
                                  int num_tokens, int D, int H) {
    matmul_gate_up(
        g_quantized_x,
        g_packed_shared_gate, g_packed_shared_up,
        w.sh_gate, w.sh_up,
        g_shared_gate_out, g_shared_up_out,
        num_tokens, D, H);

    for (int t = 0; t < num_tokens; ++t) {
        const float scale_g = g_scale_shared_gate * g_x_scale[t];
        const float scale_u = g_scale_shared_up * g_x_scale[t];
        float* h_row = g_shared_gated_fp32 + (size_t)t * H;
        int8_t* hq_row = g_shared_quantized_gated + (size_t)t * H;
        float max_abs = 0.0f;

        // 生成 SwiGLU hidden 的同时统计 max_abs，避免量化函数再次扫描一遍。
        for (int h = 0; h < H; ++h) {
            const float gate =
                (float)g_shared_gate_out[(size_t)t * H + h] * scale_g;
            const float up =
                (float)g_shared_up_out[(size_t)t * H + h] * scale_u;
            const float value = (gate / (1.0f + expf(-gate))) * up;
            h_row[h] = value;
            max_abs = std::max(max_abs, fabsf(value));
        }
        g_shared_gated_scale[t] =
            quantize_hidden_row(h_row, hq_row, H, max_abs);
    }

    matmul_small_m_or_packed(
        g_shared_quantized_gated, g_packed_shared_down, w.sh_down,
        g_shared_down_out, num_tokens, H, D);

    for (int t = 0; t < num_tokens; ++t) {
        const float dequant = g_shared_gated_scale[t] * g_scale_shared_down;
        float* y_t = y + (size_t)t * D;
        const int32_t* out_t = g_shared_down_out + (size_t)t * D;
#if defined(__AVX512F__)
        const __m512 dequant_vec = _mm512_set1_ps(dequant);
        for (int d = 0; d < D; d += 16) {
            const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
            const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
            const __m512 contribution = _mm512_mul_ps(out_fp32, dequant_vec);
            const __m512 y_vec = _mm512_loadu_ps(y_t + d);
            _mm512_storeu_ps(y_t + d, _mm512_add_ps(y_vec, contribution));
        }
#else
        for (int d = 0; d < D; ++d) {
            y_t[d] += (float)out_t[d] * dequant;
        }
#endif
    }
}

/**
 * @brief 执行完整的 DeepSeek-V3 风格 W8A8 MoE 前向计算。
 *
 * 功能：输出公式为
 *     y_t = x_t + shared_expert(x_t)
 *           + sum_{e in TopK(t)} gate[t,e] * expert_e(x_t)。
 * Router 使用 FP32；共享专家和路由专家的三个线性投影使用 INT8 权重、INT8
 * 激活和 INT32 累加；SiLU、scale 乘法以及最终专家加权在 FP32 中完成。
 *
 * 实现思路：
 * 1. memcpy 初始化残差 y=x；
 * 2. 计算 Router logit、Sigmoid、带 bias 的 Top-K 和无 bias 路由权重；
 * 3. 输入按 token 量化一次，共享专家与路由专家复用 x_q/s_x；
 * 4. 先批量计算共享专家；
 * 5. 按专家遍历，将该专家命中的 token Gather 到紧凑 g_amx_A；
 * 6. 对聚拢后的 token 批量执行 Gate/Up AMX GEMM、SwiGLU 和隐藏重量化；
 * 7. 执行 Down AMX GEMM，再按对应路由权重反量化并累加到原 token 输出。
 *
 * 向量化情况：
 * - Router 点积和输入量化在可用时使用 AVX-512；
 * - Gate/Up 使用融合 AMX；M<=4 的投影切换到 AVX-512BW 专用 kernel；
 * - Top-K、Sigmoid/SwiGLU 和分发保持标量；Down 反量化/加权使用 AVX-512；
 * - 未支持 AVX-512/AMX 的构建自动使用等价标量路径。
 *
 * @param x          FP32 输入，布局 [num_tokens,D]。
 * @param w          模型维度、Router、专家 INT8 权重及反量化 scale。
 * @param y          FP32 输出，布局 [num_tokens,D]。
 * @param num_tokens 当前批次 token 数量。
 */
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const int K = w.top_k;

    std::memcpy(y, x, (size_t)num_tokens * D * sizeof(float));

    compute_logits(x, w, num_tokens, D, E);
    compute_affinity_and_topk(w, num_tokens, E, K);
    compute_topk_weights(num_tokens, K);
    dispatch_tokens_to_experts(num_tokens, E, K);
    quantize_input(x, num_tokens, D);

    compute_shared_expert(w, y, num_tokens, D, H);

    for (int e = 0; e < E; ++e) {
        const int count = g_expert_token_count[e];
        if (count == 0) continue;

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            std::memcpy(g_amx_A + (size_t)i * D,
                        g_quantized_x + (size_t)t * D,
                        (size_t)D * sizeof(int8_t));
        }

        const size_t expert_elems = (size_t)D * H;
        const int8_t* w_gate_e = w.w_gate + (size_t)e * expert_elems;
        const int8_t* w_up_e = w.w_up + (size_t)e * expert_elems;
        const int8_t* w_down_e = w.w_down + (size_t)e * expert_elems;

        matmul_gate_up(
            g_amx_A,
            g_packed_gate[e], g_packed_up[e],
            w_gate_e, w_up_e,
            g_gate_out, g_up_out,
            count, D, H);

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            const float scale_g = g_scale_gate[e] * g_x_scale[t];
            const float scale_u = g_scale_up[e] * g_x_scale[t];
            float* h_row = g_gated_fp32 + (size_t)i * H;
            int8_t* hq_row = g_quantized_gated + (size_t)i * H;
            float max_abs = 0.0f;

            for (int h = 0; h < H; ++h) {
                const float gate =
                    (float)g_gate_out[(size_t)i * H + h] * scale_g;
                const float up =
                    (float)g_up_out[(size_t)i * H + h] * scale_u;
                const float value = (gate / (1.0f + expf(-gate))) * up;
                h_row[h] = value;
                max_abs = std::max(max_abs, fabsf(value));
            }
            g_gated_scale[i] =
                quantize_hidden_row(h_row, hq_row, H, max_abs);
        }

        matmul_small_m_or_packed(
            g_quantized_gated, g_packed_down[e], w_down_e,
            g_down_out, count, H, D);

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            const float route_weight = g_expert_token_weight[e][i];
            // Top-K 槽位已在分发时保存，后续专家并行重构时可直接定位输出槽。
            const uint8_t topk_slot = g_expert_topk_slot[e][i];
            (void)topk_slot;

            const float dequant = g_gated_scale[i] * g_scale_down[e];
            float* y_t = y + (size_t)t * D;
            const int32_t* out_t = g_down_out + (size_t)i * D;
#if defined(__AVX512F__)
            const __m512 dequant_vec = _mm512_set1_ps(dequant);
            const __m512 weight_vec = _mm512_set1_ps(route_weight);
            for (int d = 0; d < D; d += 16) {
                const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
                const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
                const __m512 dequantized =
                    _mm512_mul_ps(out_fp32, dequant_vec);
                const __m512 contribution =
                    _mm512_mul_ps(weight_vec, dequantized);
                const __m512 y_vec = _mm512_loadu_ps(y_t + d);
                _mm512_storeu_ps(y_t + d,
                                 _mm512_add_ps(y_vec, contribution));
            }
#else
            for (int d = 0; d < D; ++d) {
                y_t[d] += route_weight * ((float)out_t[d] * dequant);
            }
#endif
        }
    }
}
