// Shared dequantisation for GGML block formats on a raw byte-addressed uint buffer `w`.
// The including shader must declare `uint w[]` first and define QTYPE:
//   0 = Q4_K | 1 = Q5_K | 2 = Q6_K | 3 = Q8_0
#if QTYPE == 0
  #define BLK 256u
  #define BYTES 144u
#elif QTYPE == 1
  #define BLK 256u
  #define BYTES 176u
#elif QTYPE == 2
  #define BLK 256u
  #define BYTES 210u
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
#else
    return halfAt(b) * float(sbyteAt(b + 2u + e));
#endif
}
