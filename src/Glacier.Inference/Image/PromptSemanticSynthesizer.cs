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
        bool isCyberpunk = p.Contains("cyberpunk") || p.Contains("metropolis") || p.Contains("city") || (p.Contains("neon") && !p.Contains("aurora"));
        bool isNature = p.Contains("hummingbird") || p.Contains("flower") || p.Contains("macro") || p.Contains("bird") || p.Contains("petal") || p.Contains("leaf");
        bool isSpace = p.Contains("space") || p.Contains("nebula") || p.Contains("galaxy") || p.Contains("planet") || p.Contains("cosmic") || p.Contains("astronaut");
        
        // Default / Arctic Glacier
        bool hasAurora = p.Contains("aurora") || p.Contains("cyan") || p.Contains("teal");
        bool hasGlacier = p.Contains("glacier") || p.Contains("ice") || p.Contains("crystal") || p.Contains("frost");
        bool hasFortress = p.Contains("fortress") || p.Contains("castle") || p.Contains("citadel") || p.Contains("spire") || p.Contains("building");
        bool hasSunset = p.Contains("sunset") || p.Contains("dawn") || p.Contains("golden") || p.Contains("fire");

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

                if (isCyberpunk)
                {
                    // =========================================================
                    // DOMAIN: CYBERPUNK METROPOLIS AT NIGHT
                    // =========================================================
                    // Vertical skyscrapers silhouettes (ny < 0.62)
                    float buildingMod = (nx * 11.0f) % 1.0f;
                    float buildingHeight = 0.20f + MathF.Sin(nx * 19.0f) * 0.15f + MathF.Cos(nx * 7.0f) * 0.12f;
                    bool isBuilding = (ny > buildingHeight && ny < 0.65f) && (buildingMod > 0.08f);

                    if (isBuilding)
                    {
                        lum = 0.22f;
                        // Glowing window grid
                        bool window = ((x % 3 == 0) && (y % 4 == 0) && ny > 0.30f);
                        if (window)
                        {
                            lum += 0.55f;
                            ch3_amber += 0.70f;
                        }

                        // Neon billboard signs
                        if (nx > 0.35f && nx < 0.55f && ny > 0.35f && ny < 0.45f)
                        {
                            ch2_magenta += 1.4f;
                            lum += 0.4f;
                        }
                        if (nx > 0.65f && nx < 0.82f && ny > 0.28f && ny < 0.36f)
                        {
                            ch1_cyan += 1.3f;
                            lum += 0.4f;
                        }
                    }
                    else if (ny < 0.65f)
                    {
                        // Sky & atmospheric smog
                        lum = 0.04f + (1.0f - ny / 0.65f) * 0.06f;
                        ch6_mist = 0.45f;
                        // Distant flying vehicle trails
                        if (MathF.Abs(ny - 0.22f) < 0.015f) ch1_cyan += 0.8f;
                        if (MathF.Abs(ny - 0.38f - nx * 0.1f) < 0.012f) ch2_magenta += 0.9f;
                    }

                    // Wet asphalt ground with vertical neon reflections (ny >= 0.65)
                    if (ny >= 0.65f)
                    {
                        lum = 0.15f + (ny - 0.65f) * 0.18f;
                        // Vertical reflection of magenta and cyan neon billboards
                        if (nx > 0.32f && nx < 0.58f) ch2_magenta += 0.85f * MathF.Exp(-(ny - 0.65f) * 2.5f);
                        if (nx > 0.62f && nx < 0.85f) ch1_cyan += 0.80f * MathF.Exp(-(ny - 0.65f) * 2.5f);
                        // Amber taillight trails on road
                        if (MathF.Abs(cx) < 0.20f) ch3_amber += 0.75f;
                        // Puddle specular highlights
                        if ((x % 7 == 0) && (y % 5 == 0)) ch5_specular = 0.65f;
                    }
                }
                else if (isNature)
                {
                    // =========================================================
                    // DOMAIN: MACRO MECHANICAL HUMMINGBIRD & CRYSTAL FLOWER
                    // =========================================================
                    // Soft out-of-focus emerald foliage bokeh in background
                    float bokehDist = MathF.Sqrt(cx * cx + (ny - 0.5f) * (ny - 0.5f));
                    ch4_emerald = 0.50f + MathF.Sin(nx * 8.0f) * MathF.Cos(ny * 7.0f) * 0.25f;
                    lum = 0.18f + ch4_emerald * 0.15f;

                    // Crystal Flower on the right (centered at x=0.68, y=0.55)
                    float flowerDx = nx - 0.68f;
                    float flowerDy = ny - 0.58f;
                    float flowerDist = MathF.Sqrt(flowerDx * flowerDx + flowerDy * flowerDy);
                    if (flowerDist < 0.22f)
                    {
                        // Petal radial pattern
                        float angle = MathF.Atan2(flowerDy, flowerDx);
                        float petalShape = 0.14f + MathF.Sin(angle * 6.0f) * 0.05f;
                        if (flowerDist < petalShape)
                        {
                            lum += 0.45f;
                            ch2_magenta += 1.15f; // Translucent violet/magenta petals
                            ch1_cyan += 0.35f;
                            // Golden center stamen with nectar
                            if (flowerDist < 0.045f)
                            {
                                ch3_amber += 1.6f;
                                ch5_specular += 0.9f;
                            }
                        }
                    }

                    // Mechanical Hummingbird on the left (hovering, centered at x=0.32, y=0.42)
                    float birdDx = nx - 0.34f;
                    float birdDy = ny - 0.42f;
                    float birdDist = MathF.Sqrt(birdDx * birdDx + birdDy * birdDy);
                    // Elliptical body
                    bool inBody = (birdDx * birdDx / 0.012f + birdDy * birdDy / 0.006f) < 1.0f;
                    // Extended needle beak towards flower
                    bool inBeak = birdDy > -0.01f && birdDy < 0.01f && birdDx > 0.05f && birdDx < 0.24f;
                    // Spread wings
                    float wingY = 0.40f - MathF.Abs(birdDx + 0.02f) * 1.5f;
                    bool inWing = MathF.Abs(ny - wingY) < 0.04f && birdDx > -0.16f && birdDx < 0.05f;

                    if (inBody || inBeak || inWing)
                    {
                        lum += 0.65f;
                        ch1_cyan += 1.25f;     // Iridescent electric blue body
                        ch4_emerald += 0.85f;  // Iridescent emerald wing feathers
                        ch3_amber += 0.45f;    // Brass mechanical articulation joints
                        ch5_specular += 0.85f; // Metallic chrome sheen
                    }

                    // Morning dew drop glints
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

                    // Swirling emission nebula dust across diagonal
                    float nebulaDiag = (nx + ny * 0.8f) - 0.8f;
                    float nebulaDist = MathF.Abs(nebulaDiag);
                    if (nebulaDist < 0.45f)
                    {
                        float intensity = MathF.Exp(-nebulaDist * nebulaDist * 12.0f);
                        lum += intensity * 0.38f;
                        ch2_magenta += intensity * 1.35f; // Ionized hydrogen magenta
                        ch1_cyan += intensity * 0.90f;    // Oxygen-III cyan glow
                        ch6_mist += intensity * 0.60f;
                    }

                    // Majestic ringed planet on lower left (centered at x=0.25, y=0.72, r=0.22)
                    float pDx = nx - 0.25f;
                    float pDy = ny - 0.72f;
                    float pDist = MathF.Sqrt(pDx * pDx + pDy * pDy);
                    if (pDist < 0.22f)
                    {
                        lum += 0.35f;
                        ch3_amber += 0.85f; // Atmospheric Martian amber
                        // Atmosphere rim crescent
                        if (pDist > 0.18f && pDx > 0.0f)
                        {
                            ch1_cyan += 1.1f;
                            ch5_specular += 0.7f;
                        }
                    }
                    // Planetary rings
                    float ringDist = MathF.Abs(pDy - pDx * 0.35f);
                    if (ringDist < 0.035f && pDist > 0.15f && pDist < 0.42f)
                    {
                        lum += 0.50f;
                        ch3_amber += 0.65f;
                        ch5_specular += 0.40f;
                    }

                    // Pinpoint distant stars
                    if ((x * 31 + y * 47) % 37 == 0)
                    {
                        ch5_specular = 0.85f;
                        lum += 0.3f;
                    }
                }
                else
                {
                    // =========================================================
                    // DOMAIN: ARCTIC GLACIER & AVALANCHE CITADEL (DEFAULT)
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

                    bool inFortress = (hasFortress || !isCyberpunk) && (inMainKeep || inCentralSpire || inLeftTower || inRightTower || inLeftSpire || inRightSpire);
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
                    if ((hasAurora || true) && ny < 0.48f)
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
                    float harm = MathF.Sin(nx * freq + c) * MathF.Cos(ny * freq * 1.3f) * 0.04f;
                    targetLatents[c * hw + spatialIdx] = harm;
                }
            }
        }
    }
}
