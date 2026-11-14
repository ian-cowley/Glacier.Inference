// Shared dequantisation for GGML block formats on a raw byte-addressed uint buffer `w`.
// The including shader must declare `uint w[]` first and define QTYPE:
//   0 = Q4_K | 1 = Q5_K | 2 = Q6_K | 3 = Q8_0 | 4 = IQ4_XS | 5 = MXFP4
//   6 = Q3_K | 7 = IQ4_NL | 8 = Q2_K | 9 = Q4_0 | 10 = F16
#if QTYPE == 0
  #define BLK 256u
  #define BYTES 144u
#elif QTYPE == 1
  #define BLK 256u
  #define BYTES 176u
#elif QTYPE == 2
  #define BLK 256u
  #define BYTES 210u
#elif QTYPE == 3
  #define BLK 32u
  #define BYTES 34u
#elif QTYPE == 4
  #define BLK 256u
  #define BYTES 136u
#elif QTYPE == 5
  #define BLK 32u
  #define BYTES 17u
#elif QTYPE == 6
  #define BLK 256u
  #define BYTES 110u
#elif QTYPE == 7
  #define BLK 32u
  #define BYTES 18u
#elif QTYPE == 8
  #define BLK 256u
  #define BYTES 84u
#elif QTYPE == 9
  #define BLK 32u
  #define BYTES 18u
#elif QTYPE == 10
  #define BLK 1u
  #define BYTES 2u
#else
  #define BLK 32u
  #define BYTES 34u
#endif

uint byteAt(uint o) { return (w[o >> 2] >> ((o & 3u) * 8u)) & 0xFFu; }
int  sbyteAt(uint o) { int b = int(byteAt(o)); return b - ((b & 0x80) << 1); }
float halfAt(uint o) { return unpackHalf2x16(byteAt(o) | (byteAt(o + 1u) << 8)).x; }

void scaleMin(uint j, uint scBase, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = byteAt(scBase + j) & 63u;
        mn = byteAt(scBase + j + 4u) & 63u;
    } else {
        sc = (byteAt(scBase + j + 4u) & 0x0Fu) | ((byteAt(scBase + j - 4u) >> 6) << 4);
        mn = (byteAt(scBase + j + 4u) >> 4) | ((byteAt(scBase + j) >> 6) << 4);
    }
}

// Dequantised value of element `e` (0..BLK-1) of the block starting at byte offset `b`.
float dq(uint b, uint e) {
#if QTYPE == 0 || QTYPE == 1
    float d = halfAt(b), dmin = halfAt(b + 2u);
    uint g = e >> 6, r = e & 63u, hi = r >> 5, l = r & 31u;
    uint sc, mn;
    scaleMin(2u * g + hi, b + 4u, sc, mn);
  #if QTYPE == 0
    uint q = byteAt(b + 16u + g * 32u + l);
    uint v = hi == 0u ? (q & 15u) : (q >> 4);
  #else
    uint q = byteAt(b + 48u + g * 32u + l);
    uint v = hi == 0u ? (q & 15u) : (q >> 4);
    v += ((byteAt(b + 16u + l) >> (2u * g + hi)) & 1u) != 0u ? 16u : 0u;
  #endif
    return d * float(sc) * float(v) - dmin * float(mn);
#elif QTYPE == 2
    float d = halfAt(b + 208u);
    uint n = e >> 7, r = e & 127u, qd = r >> 5, l = r & 31u;
    uint qlB = b + n * 64u, qhB = b + 128u + n * 32u, scB = b + 192u + n * 8u;
    uint is = l >> 4;
    uint qh = byteAt(qhB + l);
    uint ql, hi2, sci;
    if (qd == 0u)      { ql = byteAt(qlB + l) & 15u;         hi2 = (qh >> 0) & 3u; sci = is + 0u; }
    else if (qd == 1u) { ql = byteAt(qlB + l + 32u) & 15u;   hi2 = (qh >> 2) & 3u; sci = is + 2u; }
    else if (qd == 2u) { ql = byteAt(qlB + l) >> 4;          hi2 = (qh >> 4) & 3u; sci = is + 4u; }
    else               { ql = byteAt(qlB + l + 32u) >> 4;    hi2 = (qh >> 6) & 3u; sci = is + 6u; }
    int q = int(ql | (hi2 << 4)) - 32;
    return d * float(sbyteAt(scB + sci)) * float(q);
#elif QTYPE == 3
    return halfAt(b) * float(sbyteAt(b + 2u + e));
#elif QTYPE == 4
    // IQ4_XS: 256 elements in 136 bytes
    const int kvalues_iq4nl[16] = int[16](-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113);
    uint sub = e >> 5;
    uint r = e & 31u;
    float d = halfAt(b);
    uint sh = byteAt(b + 2u) | (byteAt(b + 3u) << 8);
    uint sl = byteAt(b + 4u + (sub >> 1));
    uint sl_nib = (sub & 1u) == 0u ? (sl & 0xFu) : (sl >> 4);
    uint sh_bits = (sh >> (2u * sub)) & 3u;
    uint ls = sl_nib | (sh_bits << 4);
    float dl = d * float(int(ls) - 32);
    uint l = r & 15u;
    uint q = byteAt(b + 8u + sub * 16u + l);
    uint nib = (r < 16u) ? (q & 0xFu) : (q >> 4);
    return dl * float(kvalues_iq4nl[nib]);
#elif QTYPE == 5
    // MXFP4: 32 elements in 17 bytes
    const float kvalues_mxfp4[16] = float[16](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
    uint eByte = byteAt(b);
    float d = (eByte == 0u) ? 0.0 : uintBitsToFloat(eByte << 23) * 0.5;
    uint l = e & 15u;
    uint q = byteAt(b + 1u + l);
    uint nib = (e < 16u) ? (q & 0xFu) : (q >> 4);
    return d * kvalues_mxfp4[nib];
#elif QTYPE == 6
    // Q3_K: 256 elements in 110 bytes
    uint n = e >> 7;
    uint eHalf = e & 127u;
    uint j = eHalf >> 5;
    uint sub = (eHalf >> 4) & 1u;
    uint l = eHalf & 15u;
    uint isIdx = n * 8u + j * 2u + sub;

    uint idx4 = isIdx & 3u;
    uint grp = isIdx >> 2;
    uint base = (grp & 1u) == 0u ? 96u : 100u;
    uint raw = byteAt(b + base + idx4);
    uint low4 = ((grp & 2u) == 0u) ? (raw & 15u) : (raw >> 4);
    uint hi2 = (byteAt(b + 104u + idx4) >> (grp * 2u)) & 3u;
    float sc = float(int(low4 | (hi2 << 4)) - 32);

    uint qByte = byteAt(b + 32u + n * 32u + sub * 16u + l);
    uint hmByte = byteAt(b + sub * 16u + l);
    uint q2 = (qByte >> (j * 2u)) & 3u;
    uint hm = (hmByte >> (n * 4u + j)) & 1u;
    int q = int(q2) - (hm != 0u ? 0 : 4);
    return halfAt(b + 108u) * sc * float(q);
#elif QTYPE == 7
    // IQ4_NL: 32 elements in 18 bytes
    const int kvalues_iq4nl[16] = int[16](-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113);
    uint l = e & 15u;
    uint q = byteAt(b + 2u + l);
    uint nib = (e < 16u) ? (q & 15u) : (q >> 4);
    return halfAt(b) * float(kvalues_iq4nl[nib]);
#elif QTYPE == 8
    // Q2_K: 256 elements in 84 bytes
    uint step = e >> 7;
    uint eHalf = e & 127u;
    uint j = eHalf >> 5;
    uint sub = (eHalf >> 4) & 1u;
    uint l = eHalf & 15u;
    uint isIdx = step * 8u + j * 2u + sub;
    uint scByte = byteAt(b + isIdx);
    uint sc = scByte & 15u;
    uint mn = scByte >> 4;
    uint qByte = byteAt(b + 16u + step * 32u + sub * 16u + l);
    uint qVal = (qByte >> (j * 2u)) & 3u;
    float d = halfAt(b + 80u);
    float dmin = halfAt(b + 82u);
    return d * float(sc) * float(qVal) - dmin * float(mn);
#elif QTYPE == 9
    // Q4_0: 32 elements in 18 bytes
    uint l = e & 15u;
    uint q = byteAt(b + 2u + l);
    int qVal = int((e < 16u) ? (q & 15u) : (q >> 4)) - 8;
    return halfAt(b) * float(qVal);
#elif QTYPE == 10
    // F16: 16-bit half float
    return halfAt(b + e * 2u);
#else
    return halfAt(b) * float(sbyteAt(b + 2u + e));
#endif
}
