#!/bin/bash
# Compiles the Vulkan compute shaders to SPIR-V (requires glslc). Output .spv files are embedded in Glacier.Inference.dll.
# Run from this directory after editing any .comp / .glsl file, then commit the regenerated .spv files.
set -e
G="glslc --target-env=vulkan1.1 -O -I."
for t in 0 1 2 3; do
  $G -DQTYPE=$t qmatvec.comp -o qmatvec_q$t.spv
  $G -DMOE -DQTYPE=$t qmatvec.comp -o qmatvec_moe_q$t.spv
  $G -DQTYPE=$t embed.comp -o embed_q$t.spv
done
$G -DOP=0 elementwise.comp -o elementwise_swiglu.spv
$G -DOP=1 elementwise.comp -o elementwise_add.spv
for s in rmsnorm gdn_conv gdn_l2 gdn_rec f32_matvec moe_topk moe_combine attn_prep attn_decode; do $G $s.comp -o $s.spv; done
ls *.spv
