#!/bin/bash
# Compiles the Vulkan compute shaders to SPIR-V (requires glslc). Output .spv files are embedded in Glacier.Inference.dll.
# Run from this directory after editing any .comp / .glsl file, then commit the regenerated .spv files.
set -e
G="glslc --target-env=vulkan1.1 -O -I."
for t in 0 1 2 3 4 5 6 7 8 9 10 11; do
  $G -DQTYPE=$t qmatvec.comp -o qmatvec_q$t.spv
  $G -DMOE -DQTYPE=$t qmatvec.comp -o qmatvec_moe_q$t.spv
  $G -DQTYPE=$t qmatmat.comp -o qmatmat_q$t.spv
  $G -DQTYPE=$t embed.comp -o embed_q$t.spv
  $G -DBATCH -DQTYPE=$t embed.comp -o embed_batch_q$t.spv
done
$G -DOP=0 elementwise.comp -o elementwise_swiglu.spv
$G -DOP=1 elementwise.comp -o elementwise_add.spv
$G -DBATCH gdn_conv.comp -o gdn_conv_b.spv
$G -DBATCH gdn_l2.comp -o gdn_l2_b.spv
$G -DBATCH gdn_rec.comp -o gdn_rec_b.spv
for s in rmsnorm gdn_conv gdn_l2 gdn_rec f32_matvec f32_matmat bias_add_batch copy_f32 moe_topk moe_combine attn_prep attn_decode trellis_pre_rht trellis_gemv trellis_gemv_splitk trellis_post_rht trellis_gather_f32; do $G $s.comp -o $s.spv; done
$G -DPRE_RHT_128 trellis_pre_rht.comp -o trellis_pre_rht_128.spv
$G -DPOST_RHT_128 trellis_post_rht.comp -o trellis_post_rht_128.spv
G13="glslc --target-env=vulkan1.3 -O -I."
$G13 -DUSE_DP4A trellis_gemv.comp -o trellis_gemv_dp4a.spv
$G13 -DUSE_DP4A trellis_gemv_splitk.comp -o trellis_gemv_splitk_dp4a.spv
ls *.spv
