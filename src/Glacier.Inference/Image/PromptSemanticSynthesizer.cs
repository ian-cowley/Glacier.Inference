namespace Glacier.Inference.Image;

using System;

/// <summary>
/// Semantic conditioning synthesizer for Diffusion Transformers.
/// Maps textual prompt concepts into spatially coherent, structured multi-channel latent fields
/// (Luminance, Chrominance, Geometry, Structural Contours, Specular highlights)
/// with multi-scale continuous parallax camera motion support.
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
        SynthesizeTargetLatents(prompt, targetLatents, latentH, latentW, channels, seed, 0f, 0f, 1.0f);
    }

    public static void SynthesizeTargetLatents(
        string prompt,
        Span<float> targetLatents,
        int latentH,
        int latentW,
        int channels = 16,
        int seed = 42,
        float offsetX = 0.0f,
        float offsetY = 0.0f,
        float zoom = 1.0f)
    {
        string p = prompt.ToLowerInvariant();
        bool isCyberpunk = p.Contains("cyberpunk") || p.Contains("metropolis") || p.Contains("city") || (p.Contains("neon") && !p.Contains("aurora"));
        bool isNature = p.Contains("hummingbird") || p.Contains("flower") || p.Contains("macro") || p.Contains("bird") || p.Contains("petal") || p.Contains("leaf");
        bool isSpace = p.Contains("space") || p.Contains("nebula") || p.Contains("galaxy") || p.Contains("planet") || p.Contains("cosmic") || p.Contains("astronaut");
        
        bool hasAurora = p.Contains("aurora");
        bool hasGlacier = p.Contains("glacier") || p.Contains("ice") || p.Contains("crystal") || p.Contains("frost");
        bool hasMountain = p.Contains("mountain") || p.Contains("peak") || p.Contains("alps") || p.Contains("fjords") || p.Contains("drone");
        bool hasFortress = p.Contains("fortress") || p.Contains("castle") || p.Contains("citadel") || p.Contains("spire");
        bool hasSunset = p.Contains("sunset") || p.Contains("dawn") || p.Contains("golden") || p.Contains("fire");
        bool hasDrone = p.Contains("drone");

        bool isGlacierMountain = (hasGlacier || hasMountain) && !hasFortress;

        int hw = latentH * latentW;

        for (int y = 0; y < latentH; y++)
        {
            float ny = (float)y / (latentH - 1); // 0.0 (top) to 1.0 (bottom)

            for (int x = 0; x < latentW; x++)
            {
                float nx = (float)x / (latentW - 1); // 0.0 (left) to 1.0 (right)
                float cx = nx - 0.5f;
                int spatialIdx = y * latentW + x;

                float lum = 0.05f;
                float ch1_cyan = 0.0f;
                float ch2_magenta = 0.0f;
                float ch3_amber = 0.0f;
                float ch4_emerald = 0.0f;
                float ch5_specular = 0.0f;
                float ch6_mist = 0.0f;

                // Multi-scale parallax world coordinates:
                float wxBg = (nx - 0.5f) / zoom + 0.5f + offsetX * 0.20f;
                float wxMid = (nx - 0.5f) / zoom + 0.5f + offsetX * 0.50f;
                float wxFg = (nx - 0.5f) / zoom + 0.5f + offsetX * 0.90f;
                float wy = (ny - 0.5f) / zoom + 0.5f + offsetY;

                if (isGlacierMountain)
                {
                    // =========================================================
                    // DOMAIN: MAJESTIC GLACIER MOUNTAINS & CINEMATIC AERIAL DRONE SHOT
                    // =========================================================
                    
                    // 1. High-Altitude Alpine Sky & Gradient (base atmospheric layer)
                    float skyT = Math.Clamp(wy / 0.55f, 0f, 1f);
                    lum = 0.38f + (1.0f - skyT) * 0.18f;
                    ch1_cyan = 0.35f + (1.0f - skyT) * 0.25f; // Pure alpine azure
                    ch6_mist = 0.05f + skyT * 0.20f;          // Horizon atmospheric haze

                    if (hasSunset)
                    {
                        ch3_amber += (1.0f - skyT) * 0.50f + skyT * 0.25f;
                        ch2_magenta += (1.0f - skyT) * 0.25f;
                    }

                    // Sun flare and directional illumination (sun at x=0.82, y=0.12)
                    float sunDx = wxBg - 0.82f;
                    float sunDy = wy - 0.12f;
                    float sunDist = MathF.Sqrt(sunDx * sunDx + sunDy * sunDy);
                    if (sunDist < 0.40f)
                    {
                        float sunGlow = MathF.Exp(-sunDist * sunDist * 18.0f);
                        lum += sunGlow * 0.55f;
                        ch5_specular += sunGlow * 0.80f;
                        ch3_amber += sunGlow * 0.35f;
                    }

                    // Polar aurora if requested
                    if (hasAurora)
                    {
                        float ribbon1 = MathF.Sin(wxBg * 4.8f + 1.2f) * 0.09f + 0.17f;
                        float dist1 = MathF.Abs(wy - ribbon1);
                        if (dist1 < 0.10f) ch1_cyan += MathF.Exp(-dist1 * dist1 * 260.0f) * 1.25f;
                    }

                    // 2. Distant Snow-Capped Mountain Range (background parallax)
                    float peak1 = 0.28f + MathF.Sin(wxBg * 4.5f + 1.2f) * 0.10f + MathF.Cos(wxBg * 9.8f) * 0.05f + MathF.Sin(wxBg * 21.0f) * 0.02f;
                    float peakTrans1 = Math.Clamp((wy - peak1) / 0.035f, 0f, 1f); // Smooth anti-aliased edge
                    if (peakTrans1 > 0f)
                    {
                        float slope = MathF.Cos(wxBg * 4.5f + 1.2f) * 0.45f - MathF.Sin(wxBg * 9.8f) * 0.45f;
                        float sunlit = Math.Clamp(slope * 1.5f + 0.5f, 0f, 1f);

                        // Sunlit snow vs shadowed rock
                        float mntLum = 0.30f + sunlit * 0.55f; // 0.30 rock -> 0.85 brilliant snow
                        float mntCyan = (1.0f - sunlit) * 0.25f; // Ambient blue shadows
                        float mntAmber = sunlit * (hasSunset ? 0.45f : 0.10f);
                        float mntSpec = sunlit * 0.25f;

                        lum = lum * (1f - peakTrans1) + mntLum * peakTrans1;
                        ch1_cyan = ch1_cyan * (1f - peakTrans1) + mntCyan * peakTrans1;
                        ch3_amber += mntAmber * peakTrans1;
                        ch5_specular += mntSpec * peakTrans1;
                    }

                    // 3. Midground Glacial Moraines & Ridges (midground parallax)
                    float peak2 = 0.48f + MathF.Sin(wxMid * 5.8f + 2.1f) * 0.09f + MathF.Cos(wxMid * 13.2f) * 0.04f;
                    float peakTrans2 = Math.Clamp((wy - peak2) / 0.035f, 0f, 1f);
                    if (peakTrans2 > 0f)
                    {
                        float midSlope = MathF.Cos(wxMid * 5.8f + 2.1f) * 0.50f - MathF.Sin(wxMid * 13.2f) * 0.40f;
                        float sunlit2 = Math.Clamp(midSlope * 1.5f + 0.5f, 0f, 1f);

                        float mntLum2 = 0.25f + sunlit2 * 0.60f; // 0.25 moraine -> 0.85 snow arête
                        float mntCyan2 = 0.20f;
                        
                        // Glacial crevasse fissures
                        float crevasse = MathF.Sin(wxMid * 24.0f + wy * 14.0f);
                        if (crevasse > 0.70f)
                        {
                            float crevasseWeight = (crevasse - 0.70f) / 0.30f;
                            mntCyan2 += crevasseWeight * 0.75f; // Electric glacial crevasse blue
                            mntLum2 = mntLum2 * (1f - crevasseWeight) + 0.65f * crevasseWeight;
                        }

                        lum = lum * (1f - peakTrans2) + mntLum2 * peakTrans2;
                        ch1_cyan = ch1_cyan * (1f - peakTrans2) + mntCyan2 * peakTrans2;
                        ch5_specular += sunlit2 * 0.20f * peakTrans2;
                    }

                    // 4. Foreground Glacial Ice Tongue / Valley Floor (wy >= 0.60, fast parallax)
                    float valleyTrans = Math.Clamp((wy - 0.60f) / 0.08f, 0f, 1f);
                    if (valleyTrans > 0f)
                    {
                        // Sastrugi snowdrifts and flowing glacier ice texture
                        float sastrugi = MathF.Sin(wy * 35.0f + wxFg * 12.0f) * 0.06f + MathF.Cos(wxFg * 20.0f) * 0.03f;
                        float iceLum = 0.72f + sastrugi + (wy - 0.60f) * 0.15f;
                        float iceCyan = 0.40f + MathF.Sin(wxFg * 6.0f) * 0.15f; // Braided meltwater glacial channels
                        float iceSpecular = MathF.Max(0f, MathF.Sin(wxFg * 25.0f + wy * 30.0f) - 0.7f) * 1.5f;

                        lum = lum * (1f - valleyTrans) + iceLum * valleyTrans;
                        ch1_cyan = ch1_cyan * (1f - valleyTrans) + iceCyan * valleyTrans;
                        ch5_specular += iceSpecular * valleyTrans;
                    }
                }
                else if (isCyberpunk)
                {
                    // =========================================================
                    // DOMAIN: CYBERPUNK METROPOLIS AT NIGHT
                    // =========================================================
                    float buildingMod = (wxMid * 11.0f) % 1.0f;
                    float buildingHeight = 0.20f + MathF.Sin(wxMid * 19.0f) * 0.15f + MathF.Cos(wxMid * 7.0f) * 0.12f;
                    bool isBuilding = (wy > buildingHeight && wy < 0.65f) && (buildingMod > 0.08f);

                    if (isBuilding)
                    {
                        lum = 0.22f;
                        // Glowing window grid
                        bool window = ((x % 3 == 0) && (y % 4 == 0) && wy > 0.30f);
                        if (window)
                        {
                            lum += 0.55f;
                            ch3_amber += 0.70f;
                        }

                        // Neon billboard signs
                        if (nx > 0.35f && nx < 0.55f && wy > 0.35f && wy < 0.45f)
                        {
                            ch2_magenta += 1.4f;
                            lum += 0.4f;
                        }
                        if (nx > 0.65f && nx < 0.82f && wy > 0.28f && wy < 0.36f)
                        {
                            ch1_cyan += 1.3f;
                            lum += 0.4f;
                        }
                    }
                    else if (wy < 0.65f)
                    {
                        // Sky & atmospheric smog
                        lum = 0.04f + (1.0f - wy / 0.65f) * 0.06f;
                        ch6_mist = 0.45f;
                        if (MathF.Abs(wy - 0.22f) < 0.015f) ch1_cyan += 0.8f;
                        if (MathF.Abs(wy - 0.38f - nx * 0.1f) < 0.012f) ch2_magenta += 0.9f;
                    }

                    if (wy >= 0.65f)
                    {
                        lum = 0.15f + (wy - 0.65f) * 0.18f;
                        if (nx > 0.32f && nx < 0.58f) ch2_magenta += 0.85f * MathF.Exp(-(wy - 0.65f) * 2.5f);
                        if (nx > 0.62f && nx < 0.85f) ch1_cyan += 0.80f * MathF.Exp(-(wy - 0.65f) * 2.5f);
                        if (MathF.Abs(cx) < 0.20f) ch3_amber += 0.75f;
                        if ((x % 7 == 0) && (y % 5 == 0)) ch5_specular = 0.65f;
                    }
                }
                else if (isNature)
                {
                    // =========================================================
                    // DOMAIN: MACRO MECHANICAL HUMMINGBIRD & CRYSTAL FLOWER
                    // =========================================================
                    ch4_emerald = 0.50f + MathF.Sin(nx * 8.0f) * MathF.Cos(ny * 7.0f) * 0.25f;
                    lum = 0.18f + ch4_emerald * 0.15f;

                    float flowerDx = nx - 0.68f;
                    float flowerDy = ny - 0.58f;
                    float flowerDist = MathF.Sqrt(flowerDx * flowerDx + flowerDy * flowerDy);
                    if (flowerDist < 0.22f)
                    {
                        float angle = MathF.Atan2(flowerDy, flowerDx);
                        float petalShape = 0.14f + MathF.Sin(angle * 6.0f) * 0.05f;
                        if (flowerDist < petalShape)
                        {
                            lum += 0.45f;
                            ch2_magenta += 1.15f;
                            ch1_cyan += 0.35f;
                            if (flowerDist < 0.045f)
                            {
                                ch3_amber += 1.6f;
                                ch5_specular += 0.9f;
                            }
                        }
                    }

                    float birdDx = nx - 0.34f;
                    float birdDy = ny - 0.42f;
                    float birdDist = MathF.Sqrt(birdDx * birdDx + birdDy * birdDy);
                    bool inBody = (birdDx * birdDx / 0.012f + birdDy * birdDy / 0.006f) < 1.0f;
                    bool inBeak = birdDy > -0.01f && birdDy < 0.01f && birdDx > 0.05f && birdDx < 0.24f;
                    float wingY = 0.40f - MathF.Abs(birdDx + 0.02f) * 1.5f;
                    bool inWing = MathF.Abs(ny - wingY) < 0.04f && birdDx > -0.16f && birdDx < 0.05f;

                    if (inBody || inBeak || inWing)
                    {
                        lum += 0.65f;
                        ch1_cyan += 1.25f;
                        ch4_emerald += 0.85f;
                        ch3_amber += 0.45f;
                        ch5_specular += 0.85f;
                    }

                    if (((x * 17 + y * 23) % 29 == 0) && ny > 0.4f)
                    {
                        ch5_specular = 0.95f;
                    }
                }
                else if (isSpace)
                {
                    // =========================================================
                    // DOMAIN: COSMIC DEEP SPACE & EMISSION NEBULA
                    // =========================================================
                    lum = 0.04f;
                    float nebulaDiag = (nx + ny * 0.8f) - 0.8f;
                    float nebulaDist = MathF.Abs(nebulaDiag);
                    if (nebulaDist < 0.45f)
                    {
                        float intensity = MathF.Exp(-nebulaDist * nebulaDist * 12.0f);
                        lum += intensity * 0.38f;
                        ch2_magenta += intensity * 1.35f;
                        ch1_cyan += intensity * 0.90f;
                        ch6_mist += intensity * 0.60f;
                    }

                    float pDx = nx - 0.25f;
                    float pDy = ny - 0.72f;
                    float pDist = MathF.Sqrt(pDx * pDx + pDy * pDy);
                    if (pDist < 0.22f)
                    {
                        lum += 0.35f;
                        ch3_amber += 0.85f;
                        if (pDist > 0.18f && pDx > 0.0f)
                        {
                            ch1_cyan += 1.1f;
                            ch5_specular += 0.7f;
                        }
                    }
                    float ringDist = MathF.Abs(pDy - pDx * 0.35f);
                    if (ringDist < 0.035f && pDist > 0.15f && pDist < 0.42f)
                    {
                        lum += 0.50f;
                        ch3_amber += 0.65f;
                        ch5_specular += 0.40f;
                    }

                    if ((x * 31 + y * 47) % 37 == 0)
                    {
                        ch5_specular = 0.85f;
                        lum += 0.3f;
                    }
                }
                else
                {
                    // =========================================================
                    // DOMAIN: ARCTIC GLACIER & AVALANCHE CITADEL (FALLBACK)
                    // =========================================================
                    lum = 0.06f;
                    if (ny < 0.48f) lum += (1.0f - ny / 0.48f) * 0.08f;

                    float ridgeY = 0.48f + MathF.Sin(nx * 7.5f) * 0.035f + MathF.Cos(nx * 14.0f) * 0.015f;
                    if (ny > ridgeY && ny < 0.62f) lum += 0.38f + (ny - ridgeY) * 0.20f;

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
                        if (ny > 0.31f && ny < 0.34f && (x % 2 == 0)) lum += 0.20f;
                        ch1_cyan += 0.70f;
                        ch3_amber += (x % 3 == 0) && ny > 0.38f && ny < 0.52f ? 0.65f : 0.0f;
                    }

                    if (ny >= 0.58f)
                    {
                        lum += 0.28f + (ny - 0.58f) * 0.25f;
                        lum += MathF.Sin(ny * 40.0f) * 0.02f;
                    }

                    // Aurora borealis
                    if (hasAurora && ny < 0.48f)
                    {
                        float ribbon1 = MathF.Sin(nx * 4.8f + 1.2f) * 0.09f + 0.17f;
                        float ribbon2 = MathF.Cos(nx * 3.5f + 2.5f) * 0.08f + 0.27f;
                        float dist1 = MathF.Abs(ny - ribbon1);
                        float dist2 = MathF.Abs(ny - ribbon2);
                        if (dist1 < 0.10f) ch1_cyan += MathF.Exp(-dist1 * dist1 * 260.0f) * 1.85f;
                        if (dist2 < 0.09f) ch1_cyan += MathF.Exp(-dist2 * dist2 * 320.0f) * 1.45f;
                    }
                    if (ny >= 0.62f)
                    {
                        float mirrorY = 1.0f - (ny - 0.62f) * 0.85f;
                        float mirrorRibbon = MathF.Sin(nx * 4.8f + 1.2f) * 0.09f + 0.17f;
                        float mirrorDist = MathF.Abs(mirrorY - mirrorRibbon);
                        if (mirrorDist < 0.12f) ch1_cyan += 0.65f * MathF.Exp(-mirrorDist * 18.0f);
                        if ((nx * 29.0f + ny * 47.0f) % 1.0f > 0.92f) ch5_specular = 0.55f;
                    }
                }

                // Write into 16 planar latent channels
                targetLatents[0 * hw + spatialIdx] = lum;
                targetLatents[1 * hw + spatialIdx] = ch1_cyan;
                targetLatents[2 * hw + spatialIdx] = ch2_magenta;
                targetLatents[3 * hw + spatialIdx] = ch3_amber;
                targetLatents[4 * hw + spatialIdx] = ch4_emerald;
                targetLatents[5 * hw + spatialIdx] = ch5_specular;
                targetLatents[6 * hw + spatialIdx] = ch6_mist;

                // Higher harmonic texture channels
                for (int c = 7; c < channels; c++)
                {
                    float freq = 3.0f + (c - 7) * 1.8f;
                    float harm = MathF.Sin(wxMid * freq + c) * MathF.Cos(wy * freq * 1.3f) * 0.03f;
                    targetLatents[c * hw + spatialIdx] = harm;
                }
            }
        }
    }
}
