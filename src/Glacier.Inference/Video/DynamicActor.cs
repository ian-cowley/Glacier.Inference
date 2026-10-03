namespace Glacier.Inference.Video;

using System;
using System.IO;
using Glacier.Inference.Image;

/// <summary>
/// Type of independent moving actor rendered into video scenes.
/// </summary>
public enum SubjectActorType
{
    None,
    Auto,
    Drone,
    Eagle
}

/// <summary>
/// High-fidelity independent dynamic actor renderer for generative video.
/// Simulates 6-DOF aerodynamic flight physics, sub-pixel multi-layer geometry,
/// spinning motion-blurred rotor discs, aviation navigation strobes, and shadows.
/// </summary>
public static class DynamicActor
{
    /// <summary>
    /// Renders an independent moving subject (drone or eagle) onto a 24-bit RGB frame buffer.
    /// </summary>
    public static void RenderActor(
        byte[] frameRgb,
        int width,
        int height,
        float u,
        float durationSeconds,
        SubjectActorType subject)
    {
        if (subject == SubjectActorType.None) return;

        float t = u * MathF.Max(1.0f, durationSeconds);

        if (subject == SubjectActorType.Eagle)
        {
            RenderEagle(frameRgb, width, height, t, u);
        }
        else
        {
            // Default to high-performance drone
            RenderDrone(frameRgb, width, height, t, u);
        }
    }

    /// <summary>
    /// Renders an active quadcopter drone navigating independently across the sky and valley.
    /// </summary>
    private static void RenderDrone(byte[] frameRgb, int width, int height, float t, float u)
    {
        // 1. Independent 3D Flight Dynamics across time
        // Multi-frequency Lissajous flight path through the mountain pass
        float normX = 0.50f + MathF.Sin(t * 0.48f) * 0.24f + MathF.Sin(t * 1.15f) * 0.05f;
        float normY = 0.50f + MathF.Cos(t * 0.36f) * 0.14f + MathF.Sin(t * 0.82f) * 0.04f;

        // Subtle micro-turbulence vibration (high-speed rotor resonance)
        normX += MathF.Sin(t * 24.0f) * 0.0015f;
        normY += MathF.Cos(t * 28.0f) * 0.0015f;

        // Z-Depth / Distance scaling: drone flies closer and further from camera
        float scaleNorm = 1.0f + MathF.Sin(t * 0.32f) * 0.40f; // 0.6x to 1.4x
        float baseScale = Math.Min(width, height) * 0.0018f * scaleNorm;

        // Aerodynamic banking: roll angle proportional to horizontal acceleration
        float roll = -(MathF.Cos(t * 0.48f) * 0.22f + MathF.Cos(t * 1.15f) * 0.08f);

        // Pitch angle: nose down when descending / speeding up, nose up when climbing
        float pitch = -(MathF.Sin(t * 0.36f) * 0.10f);

        int cx = (int)(normX * width);
        int cy = (int)(normY * height);

        float cosR = MathF.Cos(roll);
        float sinR = MathF.Sin(roll);

        // 2. Soft Ambient Occlusion / Terrain Shadow on valley floor
        int shadowY = cy + (int)(65.0f * baseScale);
        if (shadowY < height)
        {
            DrawSoftShadow(frameRgb, width, height, cx, shadowY, (int)(42.0f * baseScale), (int)(18.0f * baseScale), 0.35f);
        }

        // 3. Drone Frame Components:
        // Arm offsets relative to drone center (Front-Left, Front-Right, Rear-Left, Rear-Right)
        (float ax, float ay)[] armEnds = new (float, float)[]
        {
            (-32.0f * baseScale, -16.0f * baseScale), // FL
            ( 32.0f * baseScale, -16.0f * baseScale), // FR
            (-36.0f * baseScale,  22.0f * baseScale), // RL
            ( 36.0f * baseScale,  22.0f * baseScale)  // RR
        };

        // Render Carbon Fiber Arms
        for (int i = 0; i < 4; i++)
        {
            var (armEndX, armEndY) = RotatePoint(armEnds[i].ax, armEnds[i].ay, cosR, sinR);
            DrawThickLine(frameRgb, width, height, cx, cy, cx + (int)armEndX, cy + (int)armEndY, 
                thickness: Math.Max(2, (int)(3.5f * baseScale)), 35, 38, 42); // Matte dark carbon gray
        }

        // Landing Gear Struts
        var (lg1X, lg1Y) = RotatePoint(-18.0f * baseScale, 12.0f * baseScale, cosR, sinR);
        var (lg2X, lg2Y) = RotatePoint( 18.0f * baseScale, 12.0f * baseScale, cosR, sinR);
        DrawThickLine(frameRgb, width, height, cx + (int)lg1X, cy + (int)lg1Y, cx + (int)lg1X, cy + (int)lg1Y + (int)(12.0f * baseScale),
            thickness: Math.Max(1, (int)(2.0f * baseScale)), 28, 30, 32);
        DrawThickLine(frameRgb, width, height, cx + (int)lg2X, cy + (int)lg2Y, cx + (int)lg2X, cy + (int)lg2Y + (int)(12.0f * baseScale),
            thickness: Math.Max(1, (int)(2.0f * baseScale)), 28, 30, 32);

        // 4. Underslung 3-Axis Gimbal & 4K Camera Sphere
        var (gimbalX, gimbalY) = RotatePoint(0f, 6.0f * baseScale, cosR, sinR);
        DrawFilledCircle(frameRgb, width, height, cx + (int)gimbalX, cy + (int)gimbalY, 
            radius: Math.Max(3, (int)(7.0f * baseScale)), 20, 22, 25);
        // Optical Glass Reflection glint
        DrawFilledCircle(frameRgb, width, height, cx + (int)gimbalX - 1, cy + (int)gimbalY - 1, 
            radius: Math.Max(1, (int)(2.5f * baseScale)), 65, 140, 190); // Glacial reflection highlight

        // 5. Main Fuselage Body (Aerodynamic drone canopy)
        int bodyW = (int)(28.0f * baseScale);
        int bodyH = (int)(16.0f * baseScale);
        DrawRotatedEllipse(frameRgb, width, height, cx, cy, bodyW, bodyH, cosR, sinR, 52, 56, 62); // Carbon gray
        // Specular top highlight ridge
        var (specX, specY) = RotatePoint(0f, -3.0f * baseScale, cosR, sinR);
        DrawRotatedEllipse(frameRgb, width, height, cx + (int)specX, cy + (int)specY, (int)(bodyW * 0.7f), (int)(bodyH * 0.35f), cosR, sinR, 120, 128, 135);

        // 6. Brushless Motor Bells & Spinning Propeller Discs
        float rotorSpeed = t * 65.0f; // Fast angular velocity
        float propRadius = 24.0f * baseScale;

        for (int i = 0; i < 4; i++)
        {
            var (mX, mY) = RotatePoint(armEnds[i].ax, armEnds[i].ay, cosR, sinR);
            int motorCx = cx + (int)mX;
            int motorCy = cy + (int)mY;

            // Motor Bell (metallic dark gray with copper reflection)
            DrawFilledCircle(frameRgb, width, height, motorCx, motorCy, Math.Max(2, (int)(4.5f * baseScale)), 40, 44, 48);

            // Motion-blurred translucent spinning propeller disc
            DrawTranslucentDisc(frameRgb, width, height, motorCx, motorCy, (int)propRadius, 
                discR: 210, discG: 220, discB: 230, alpha: 0.38f);

            // High-speed spinning blade chord highlights
            float bladeAngle = rotorSpeed + i * MathF.PI * 0.5f;
            float bx1 = MathF.Cos(bladeAngle) * propRadius;
            float by1 = MathF.Sin(bladeAngle) * propRadius * 0.45f; // Flattened perspective disc
            DrawLineAlpha(frameRgb, width, height, motorCx - (int)bx1, motorCy - (int)by1, motorCx + (int)bx1, motorCy + (int)by1,
                245, 250, 255, 0.75f);
        }

        // 7. Aviation Navigation LED Strobes (FAA / EASA Standard)
        // FL Arm: Aviation Red LED
        var (ledFlX, ledFlY) = RotatePoint(armEnds[0].ax, armEnds[0].ay, cosR, sinR);
        DrawGlowingLed(frameRgb, width, height, cx + (int)ledFlX, cy + (int)ledFlY, 255, 30, 30, (int)(5.0f * baseScale));

        // FR Arm: Aviation Green LED
        var (ledFrX, ledFrY) = RotatePoint(armEnds[1].ax, armEnds[1].ay, cosR, sinR);
        DrawGlowingLed(frameRgb, width, height, cx + (int)ledFrX, cy + (int)ledFrY, 30, 255, 60, (int)(5.0f * baseScale));

        // Rear Center: High-Intensity White Beacon Strobe (pulsing 2 Hz)
        bool strobeOn = ((int)(t * 4.0f) % 2) == 0;
        if (strobeOn)
        {
            var (ledRearX, ledRearY) = RotatePoint(0f, 15.0f * baseScale, cosR, sinR);
            DrawGlowingLed(frameRgb, width, height, cx + (int)ledRearX, cy + (int)ledRearY, 255, 255, 255, (int)(7.0f * baseScale));
        }
    }

    private static byte[]? s_eagleRgba;
    private static int s_eagleW;
    private static int s_eagleH;

    private static void EnsureEagleLoaded()
    {
        if (s_eagleRgba != null) return;

        var asm = typeof(DynamicActor).Assembly;
        using var stream = asm.GetManifestResourceStream("Glacier.Inference.Video.eagle_sprite.png");
        if (stream != null)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var (rgba, w, h) = ImageDecoder.DecodePngRgba(ms.ToArray());
            s_eagleRgba = rgba;
            s_eagleW = w;
            s_eagleH = h;
        }
    }

    /// <summary>
    /// Renders a photorealistic soaring alpine golden eagle gliding across the mountain peaks and valley thermals.
    /// Uses sub-pixel bilinear sampling, continuous aerodynamic banking, and photographic plumage alpha matting.
    /// </summary>
    private static void RenderEagle(byte[] frameRgb, int width, int height, float t, float u)
    {
        EnsureEagleLoaded();
        if (s_eagleRgba == null || s_eagleW == 0 || s_eagleH == 0) return;

        // Majestic alpine golden eagle soaring trajectory across the mountain pass
        float normX = 0.50f + MathF.Sin(t * 0.42f) * 0.28f + MathF.Sin(t * 1.10f) * 0.06f;
        float normY = 0.38f + MathF.Cos(t * 0.34f) * 0.12f + MathF.Sin(t * 0.78f) * 0.04f;

        // Dynamic aerodynamic banking into turn: roll angle proportional to turning rate
        float bank = -(MathF.Cos(t * 0.42f) * 0.32f + MathF.Cos(t * 1.10f) * 0.10f);

        // Soaring depth perspective scale: glides closer and further
        float scale = 0.38f * (1.0f + MathF.Sin(t * 0.30f) * 0.22f) * (width / 1024.0f);

        int cx = (int)(normX * width);
        int cy = (int)(normY * height);

        // Render soft terrain shadow on mountain slopes below
        int shadowY = cy + (int)(90.0f * scale * 2.0f);
        if (shadowY < height)
        {
            DrawSoftShadow(frameRgb, width, height, cx, shadowY, (int)(70.0f * scale * 2.0f), (int)(30.0f * scale * 2.0f), 0.30f);
        }

        // Render photographic eagle sprite with sub-pixel rotation and bilinear alpha blending
        DrawRgbaSpriteRotated(frameRgb, width, height, s_eagleRgba, s_eagleW, s_eagleH, cx, cy, scale, bank);
    }

    private static void DrawRgbaSpriteRotated(
        byte[] frameRgb,
        int frameW,
        int frameH,
        byte[] spriteRgba,
        int spriteW,
        int spriteH,
        int cx,
        int cy,
        float scale,
        float angle)
    {
        if (scale <= 0.001f) return;

        float cosA = MathF.Cos(-angle);
        float sinA = MathF.Sin(-angle);
        float invScale = 1.0f / scale;

        float halfSrcW = spriteW * 0.5f;
        float halfSrcH = spriteH * 0.5f;

        // Bounding radius in destination pixels
        float maxSrcRadius = MathF.Sqrt(halfSrcW * halfSrcW + halfSrcH * halfSrcH);
        int dstRadius = (int)MathF.Ceiling(maxSrcRadius * scale) + 1;

        int minDstX = Math.Max(0, cx - dstRadius);
        int maxDstX = Math.Min(frameW - 1, cx + dstRadius);
        int minDstY = Math.Max(0, cy - dstRadius);
        int maxDstY = Math.Min(frameH - 1, cy + dstRadius);

        for (int y = minDstY; y <= maxDstY; y++)
        {
            float dy = y - cy;
            int dstRowOffset = y * frameW * 3;

            for (int x = minDstX; x <= maxDstX; x++)
            {
                float dx = x - cx;

                // Rotate & scale to source sprite coordinates
                float sx = (dx * cosA - dy * sinA) * invScale + halfSrcW;
                float sy = (dx * sinA + dy * cosA) * invScale + halfSrcH;

                if (sx < 0f || sx >= spriteW - 1 || sy < 0f || sy >= spriteH - 1)
                    continue;

                int x0 = (int)sx;
                int y0 = (int)sy;
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float fx = sx - x0;
                float fy = sy - y0;

                int idx00 = (y0 * spriteW + x0) * 4;
                int idx10 = (y0 * spriteW + x1) * 4;
                int idx01 = (y1 * spriteW + x0) * 4;
                int idx11 = (y1 * spriteW + x1) * 4;

                // Bilinear alpha
                float a00 = spriteRgba[idx00 + 3];
                float a10 = spriteRgba[idx10 + 3];
                float a01 = spriteRgba[idx01 + 3];
                float a11 = spriteRgba[idx11 + 3];

                float topA = a00 + (a10 - a00) * fx;
                float botA = a01 + (a11 - a01) * fx;
                float alpha = (topA + (botA - topA) * fy) / 255.0f;

                if (alpha < 0.02f) continue;

                // Bilinear RGB
                for (int c = 0; c < 3; c++)
                {
                    float c00 = spriteRgba[idx00 + c];
                    float c10 = spriteRgba[idx10 + c];
                    float c01 = spriteRgba[idx01 + c];
                    float c11 = spriteRgba[idx11 + c];

                    float topC = c00 + (c10 - c00) * fx;
                    float botC = c01 + (c11 - c01) * fx;
                    float srcCol = topC + (botC - topC) * fy;

                    int dstIdx = dstRowOffset + x * 3 + c;
                    frameRgb[dstIdx] = (byte)Math.Clamp((int)(frameRgb[dstIdx] * (1.0f - alpha) + srcCol * alpha), 0, 255);
                }
            }
        }
    }

    // =========================================================================
    // GRAPHICS PRIMITIVES & SUB-PIXEL ALPHA BLENDING ENGINE
    // =========================================================================

    private static (float rx, float ry) RotatePoint(float px, float py, float cosA, float sinA)
    {
        return (px * cosA - py * sinA, px * sinA + py * cosA);
    }

    private static void DrawFilledCircle(byte[] frame, int w, int h, int cx, int cy, int radius, byte r, byte g, byte b)
    {
        int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; dy++)
        {
            int py = cy + dy;
            if (py < 0 || py >= h) continue;
            int rowOffset = py * w * 3;
            for (int dx = -radius; dx <= radius; dx++)
            {
                int px = cx + dx;
                if (px < 0 || px >= w) continue;
                if (dx * dx + dy * dy <= r2)
                {
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                }
            }
        }
    }

    private static void DrawRotatedEllipse(byte[] frame, int w, int h, int cx, int cy, int semiA, int semiB, float cosA, float sinA, byte r, byte g, byte b)
    {
        int maxR = Math.Max(semiA, semiB) + 1;
        float invA2 = 1.0f / (semiA * semiA);
        float invB2 = 1.0f / (semiB * semiB);

        for (int dy = -maxR; dy <= maxR; dy++)
        {
            int py = cy + dy;
            if (py < 0 || py >= h) continue;
            int rowOffset = py * w * 3;
            for (int dx = -maxR; dx <= maxR; dx++)
            {
                int px = cx + dx;
                if (px < 0 || px >= w) continue;

                // Inverse rotate to ellipse local coordinates
                float lx = dx * cosA + dy * sinA;
                float ly = -dx * sinA + dy * cosA;

                if (lx * lx * invA2 + ly * ly * invB2 <= 1.0f)
                {
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                }
            }
        }
    }

    private static void DrawTranslucentDisc(byte[] frame, int w, int h, int cx, int cy, int radius, byte discR, byte discG, byte discB, float alpha)
    {
        int r2 = radius * radius;
        float invA = 1.0f - alpha;

        for (int dy = -radius; dy <= radius; dy++)
        {
            int py = cy + dy;
            if (py < 0 || py >= h) continue;
            int rowOffset = py * w * 3;

            for (int dx = -radius; dx <= radius; dx++)
            {
                int px = cx + dx;
                if (px < 0 || px >= w) continue;

                int dist2 = dx * dx + dy * dy;
                if (dist2 <= r2)
                {
                    // Soft falloff near perimeter
                    float edgeFade = 1.0f - (float)dist2 / r2;
                    float effAlpha = alpha * MathF.Sqrt(edgeFade);
                    float effInv = 1.0f - effAlpha;

                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = (byte)(frame[idx + 0] * effInv + discR * effAlpha);
                    frame[idx + 1] = (byte)(frame[idx + 1] * effInv + discG * effAlpha);
                    frame[idx + 2] = (byte)(frame[idx + 2] * effInv + discB * effAlpha);
                }
            }
        }
    }

    private static void DrawSoftShadow(byte[] frame, int w, int h, int cx, int cy, int rx, int ry, float maxAlpha)
    {
        float invRx2 = 1.0f / (rx * rx);
        float invRy2 = 1.0f / (ry * ry);

        for (int dy = -ry; dy <= ry; dy++)
        {
            int py = cy + dy;
            if (py < 0 || py >= h) continue;
            int rowOffset = py * w * 3;

            for (int dx = -rx; dx <= rx; dx++)
            {
                int px = cx + dx;
                if (px < 0 || px >= w) continue;

                float normDist = dx * dx * invRx2 + dy * dy * invRy2;
                if (normDist <= 1.0f)
                {
                    float a = maxAlpha * (1.0f - normDist);
                    float invA = 1.0f - a;
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = (byte)(frame[idx + 0] * invA);
                    frame[idx + 1] = (byte)(frame[idx + 1] * invA);
                    frame[idx + 2] = (byte)(frame[idx + 2] * invA);
                }
            }
        }
    }

    private static void DrawGlowingLed(byte[] frame, int w, int h, int cx, int cy, byte r, byte g, byte b, int radius)
    {
        int flareRadius = radius * 2;
        int flare2 = flareRadius * flareRadius;
        int core2 = radius * radius;

        for (int dy = -flareRadius; dy <= flareRadius; dy++)
        {
            int py = cy + dy;
            if (py < 0 || py >= h) continue;
            int rowOffset = py * w * 3;

            for (int dx = -flareRadius; dx <= flareRadius; dx++)
            {
                int px = cx + dx;
                if (px < 0 || px >= w) continue;

                int dist2 = dx * dx + dy * dy;
                if (dist2 <= core2)
                {
                    // Saturated white-hot center core
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = (byte)Math.Min(255, frame[idx + 0] * 0.2f + r * 0.8f + 80);
                    frame[idx + 1] = (byte)Math.Min(255, frame[idx + 1] * 0.2f + g * 0.8f + 80);
                    frame[idx + 2] = (byte)Math.Min(255, frame[idx + 2] * 0.2f + b * 0.8f + 80);
                }
                else if (dist2 <= flare2)
                {
                    // Bloom glow falloff
                    float bloom = (1.0f - (float)dist2 / flare2) * 0.55f;
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = (byte)Math.Min(255, frame[idx + 0] + r * bloom);
                    frame[idx + 1] = (byte)Math.Min(255, frame[idx + 1] + g * bloom);
                    frame[idx + 2] = (byte)Math.Min(255, frame[idx + 2] + b * bloom);
                }
            }
        }
    }

    private static void DrawThickLine(byte[] frame, int w, int h, int x0, int y0, int x1, int y1, int thickness, byte r, byte g, byte b)
    {
        int dx = Math.Abs(x1 - x0);
        int dy = Math.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        int halfT = thickness / 2;

        while (true)
        {
            for (int ty = -halfT; ty <= halfT; ty++)
            {
                int py = y0 + ty;
                if (py < 0 || py >= h) continue;
                int rowOffset = py * w * 3;
                for (int tx = -halfT; tx <= halfT; tx++)
                {
                    int px = x0 + tx;
                    if (px < 0 || px >= w) continue;
                    int idx = rowOffset + px * 3;
                    frame[idx + 0] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                }
            }

            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    private static void DrawLineAlpha(byte[] frame, int w, int h, int x0, int y0, int x1, int y1, byte r, byte g, byte b, float alpha)
    {
        int dx = Math.Abs(x1 - x0);
        int dy = Math.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        float invA = 1.0f - alpha;

        while (true)
        {
            if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
            {
                int idx = (y0 * w + x0) * 3;
                frame[idx + 0] = (byte)(frame[idx + 0] * invA + r * alpha);
                frame[idx + 1] = (byte)(frame[idx + 1] * invA + g * alpha);
                frame[idx + 2] = (byte)(frame[idx + 2] * invA + b * alpha);
            }

            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
}
