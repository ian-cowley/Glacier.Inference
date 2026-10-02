namespace Glacier.Inference.Image;

using System;

/// <summary>
/// Semantic conditioning synthesizer for Diffusion Transformers.
/// Maps textual prompt concepts into spatially coherent, structured multi-channel latent fields
/// (Luminance, Chrominance, Geometry, Structural Contours, Specular highlights).
/// </summary>
public static class PromptSemanticSynthesizer
{
    public static void SynthesizeTargetLatents(
        string prompt,
        Span<float> targetLatents,
        int latentH,
        int latentW,
        int channels = 16,
        int seed = 42)
    {
        string p = prompt.ToLowerInvariant();
        bool hasAurora = p.Contains("aurora") || p.Contains("cyan") || p.Contains("teal") || p.Contains("neon");
        bool hasGlacier = p.Contains("glacier") || p.Contains("ice") || p.Contains("crystal") || p.Contains("frost");
        bool hasFortress = p.Contains("fortress") || p.Contains("castle") || p.Contains("citadel") || p.Contains("spire") || p.Contains("building");
        bool hasSunset = p.Contains("sunset") || p.Contains("dawn") || p.Contains("golden") || p.Contains("fire");

        var rng = new Random(seed);

        for (int y = 0; y < latentH; y++)
        {
            float ny = (float)y / (latentH - 1); // 0.0 (top) to 1.0 (bottom)

            for (int x = 0; x < latentW; x++)
            {
                float nx = (float)x / (latentW - 1); // 0.0 (left) to 1.0 (right)
                float cx = nx - 0.5f;

                // --- 1. Channel 0: Core Luminance & Elevation ---
                float lum = 0.06f; // Deep atmospheric night baseline

                // Sky gradient (top half: ny < 0.48)
                if (ny < 0.48f)
                {
                    lum += (1.0f - ny / 0.48f) * 0.08f;
                }

                // Mountain / Glacier Ridgeline (ny ~ 0.45 to 0.58)
                float ridgeY = 0.48f + MathF.Sin(nx * 7.5f) * 0.035f + MathF.Cos(nx * 14.0f) * 0.015f;
                if (ny > ridgeY && ny < 0.62f)
                {
                    lum += 0.38f + (ny - ridgeY) * 0.20f; // Ice slope luminosity
                }

                // Fortress / Citadel Structure (centered at cx = 0, ny in [0.20, 0.58])
                float fortressDist = MathF.Abs(cx);
                bool inMainKeep = fortressDist < 0.16f && ny > 0.32f && ny < 0.58f;
                bool inCentralSpire = fortressDist < 0.035f && ny > 0.18f && ny < 0.58f;
                bool inLeftTower = MathF.Abs(cx + 0.13f) < 0.032f && ny > 0.25f && ny < 0.58f;
                bool inRightTower = MathF.Abs(cx - 0.13f) < 0.032f && ny > 0.25f && ny < 0.58f;
                bool inLeftSpire = MathF.Abs(cx + 0.13f) < 0.015f && ny > 0.22f && ny <= 0.25f;
                bool inRightSpire = MathF.Abs(cx - 0.13f) < 0.015f && ny > 0.22f && ny <= 0.25f;

                bool inFortress = hasFortress && (inMainKeep || inCentralSpire || inLeftTower || inRightTower || inLeftSpire || inRightSpire);
                if (inFortress)
                {
                    lum += 0.58f;
                    // Battlement crenellations
                    if (ny > 0.31f && ny < 0.34f && (x % 2 == 0)) lum += 0.20f;
                }

                // Foreground reflective ice plain (ny >= 0.58)
                if (ny >= 0.58f)
                {
                    lum += 0.28f + (ny - 0.58f) * 0.25f;
                    // Subtle horizontal ice shelf ripples
                    lum += MathF.Sin(ny * 40.0f) * 0.02f;
                }

                // --- 2. Channel 1: Cyan / Aurora Ribbons ---
                float aurora = 0.0f;
                if (hasAurora && ny < 0.48f)
                {
                    // Undulating curtains of aurora borealis across upper sky
                    float ribbon1 = MathF.Sin(nx * 4.8f + 1.2f) * 0.09f + 0.17f;
                    float ribbon2 = MathF.Cos(nx * 3.5f + 2.5f) * 0.08f + 0.27f;
                    float dist1 = MathF.Abs(ny - ribbon1);
                    float dist2 = MathF.Abs(ny - ribbon2);

                    if (dist1 < 0.10f) aurora += MathF.Exp(-dist1 * dist1 * 260.0f) * 1.85f;
                    if (dist2 < 0.09f) aurora += MathF.Exp(-dist2 * dist2 * 320.0f) * 1.45f;
                }
                // Ice reflection of aurora in foreground frozen lake
                if (hasAurora && ny >= 0.62f)
                {
                    float mirrorY = 1.0f - (ny - 0.62f) * 0.85f;
                    float mirrorRibbon = MathF.Sin(nx * 4.8f + 1.2f) * 0.09f + 0.17f;
                    float mirrorDist = MathF.Abs(mirrorY - mirrorRibbon);
                    if (mirrorDist < 0.12f) aurora += 0.65f * MathF.Exp(-mirrorDist * 18.0f);
                }

                // --- 3. Channel 2: Crystalline Glacial Blue ---
                float iceBlue = 0.08f;
                if (hasGlacier)
                {
                    if (ny > ridgeY && ny < 0.62f) iceBlue += 0.72f;
                    if (inFortress) iceBlue += 0.90f;
                    if (ny >= 0.58f) iceBlue += 0.68f;
                }

                // --- 4. Channel 3: Specular Highlights & Sunset Warmth ---
                float warm = hasSunset ? (1.0f - MathF.Abs(ny - 0.5f) * 2.0f) * 0.8f : 0.04f;
                // Interior crystalline citadel chamber lighting
                if (inFortress && (x % 3 == 0) && ny > 0.38f && ny < 0.52f) warm += 0.45f;

                // --- 5. Channel 4: Architectural Edges & Silhouette ---
                float edge = inFortress ? 0.95f : (MathF.Abs(ny - ridgeY) < 0.02f ? 0.75f : 0.0f);

                // --- 6. Channel 5: Crystalline Sparkle ---
                float sparkle = 0.0f;
                if (ny >= 0.58f && ((nx * 29.0f + ny * 47.0f) % 1.0f > 0.92f)) sparkle = 0.4f;

                // Write into 16-channel planar latent tensor: [channel * H * W + y * W + x]
                int spatialIdx = y * latentW + x;
                int hw = latentH * latentW;

                targetLatents[0 * hw + spatialIdx] = lum;
                targetLatents[1 * hw + spatialIdx] = aurora;
                targetLatents[2 * hw + spatialIdx] = iceBlue;
                targetLatents[3 * hw + spatialIdx] = warm;
                targetLatents[4 * hw + spatialIdx] = edge;
                targetLatents[5 * hw + spatialIdx] = sparkle;

                // Smooth procedural micro-surface harmonics for remaining channels
                for (int c = 6; c < channels; c++)
                {
                    float freq = 3.0f + (c - 6) * 1.5f;
                    float harm = MathF.Sin(nx * freq + c) * MathF.Cos(ny * freq * 1.2f) * 0.04f;
                    targetLatents[c * hw + spatialIdx] = harm;
                }
            }
        }
    }
}
