uchar8 block(ulong val) {
    return (uchar8)(
        val,
        val >> 8,
        val >> 16,
        val >> 24,
        val >> 32,
        val >> 40,
        val >> 48,
        val >> 56
    );
}

typedef struct adder_result {
    uchar8 sum;
    uchar8 carry;
} adder_result;

adder_result half_adder(uchar8 a, uchar8 b) {
    uchar8 sum = a ^ b;
    uchar8 carry = a & b;
    return (adder_result) {
        .sum = sum,
        .carry = carry
    };
}

adder_result full_adder(uchar8 a, uchar8 b, uchar8 c) {
    uchar8 temp = a ^ b;
    uchar8 sum = temp ^ c;
    uchar8 carry = (a & b) | (temp & c);
    return (adder_result) {
        .sum = sum,
        .carry = carry
    };
}

/**
 * abc
 * d e
 * fgh
 */
typedef struct neighbor_vectors {
    uchar8 d;
    uchar8 e;

    uchar a;
    uchar b;
    uchar c;
    uchar f;
    uchar g;
    uchar h;
} neighbor_vectors;

/**
 * Takes the cell and its neighbor vectors, and returns
 * the new state of cell after one generation.
 * Each "cell" is an 8x8 region, and the "neighbors"
 * are 8x8 regions offset.
 * 
 * See https://binary-banter.github.io/game-of-life/
 * We implement the addition manually using bitwise operations
 * to emulate adders. That way, each bitplane is a separate uchar8,
 * instead of having to expand to a larger bit width (e.g., 4 bits per cell)
 */


// Shifts the 8x8 region to the Moore neighborhood, giving 8 vectors
// representing the regions shifted over.
uchar8 life1(uchar8 r, neighbor_vectors ns) {
    ushort i1 = (((ushort)ns.a & (ushort)1) << 9) | ((ushort)ns.b << 1) | ((ushort)ns.c >> 7);
    ushort8 i2 = ((convert_ushort8(ns.d) & (ushort)1) << 9) | (convert_ushort8(r) << 1) | (convert_ushort8(ns.e) >> 7);
    ushort i3 = (((ushort)ns.f & (ushort)1) << 9) | ((ushort)ns.g << 1) | ((ushort)ns.h >> 7);

    ushort8 up = shuffle(i2, (ushort8)(0,0,1,2,3,4,5,6));
    up.s0 = i1;
    ushort8 down = shuffle(i2, (ushort8)(1,2,3,4,5,6,7,7));
    down.s7 = i3;

    // abc
    // d.e
    // fgh
    uchar8 a = convert_uchar8(up >> 2);
    uchar8 b = convert_uchar8(up >> 1);
    uchar8 c = convert_uchar8(up);

    adder_result li = full_adder(a, b, c);

    uchar8 d = convert_uchar8(i2 >> 2);
    uchar8 e = convert_uchar8(i2);
    uchar8 f = convert_uchar8(down >> 2);

    adder_result mj = full_adder(d, e, f);

    uchar8 g = convert_uchar8(down >> 1);
    uchar8 h = convert_uchar8(down);

    adder_result nk = half_adder(g, h);
    adder_result yw = full_adder(li.sum, mj.sum, nk.sum);
    adder_result xz = full_adder(li.carry, mj.carry, nk.carry);

    // survive if currently alive
    uchar8 result = r;
    // born if 1,3,5,7 neighbors
    result |= yw.sum;
    // survive if 2,3,6,7 neighbors
    result &= (yw.carry ^ xz.sum);
    // survive if 0,1,2,3 neighbors
    result &= ~xz.carry;
    return result;
}


/**
 * @brief Calculates up to 8 generations of life.
 * The regions are 8x8 regions packed in 64 bits. Each region could
 * be placed anywhere in the pattern, and is linked to its 8
 * neighbors by index in the neighbors array.
 * 
 * Inputs should include all 8x8 regions with live cells or empty
 * regions that neighbor live regions.
 * 
 * @param in An array of 8x8 regions that might change.
 * @param neighbors For each item in the array, 8 indices for the neighbors of the region.
 * @param out 
 * @return kernel 
 */
kernel void sparse_life8(int gens_to_simulate, const global ulong *regions, const global ushort *neighbors, global ulong *out) {
    int id = get_global_id(0);
    uchar8 b0_0 = block(regions[neighbors[id * 8]]);
    uchar8 b1_0 = block(regions[neighbors[id * 8 + 1]]);
    uchar8 b2_0 = block(regions[neighbors[id * 8 + 2]]);
    uchar8 b0_1 = block(regions[neighbors[id * 8 + 3]]);
    uchar8 b1_1 = block(regions[id]);
    uchar8 b2_1 = block(regions[neighbors[id * 8 + 4]]);
    uchar8 b0_2 = block(regions[neighbors[id * 8 + 5]]);
    uchar8 b1_2 = block(regions[neighbors[id * 8 + 6]]);
    uchar8 b2_2 = block(regions[neighbors[id * 8 + 7]]);

    for (int gen = 0; gen < gens_to_simulate; gen++) {
        uchar8 n0_0 = life1(
            b0_0, (neighbor_vectors) {
                .a = 0, .b = 0, .c = 0,
                .d = 0,         .e = b1_0,
                .f=0,   .g=b0_1.s0,.h = b1_1.s0
            }
        );
        uchar8 n1_0 = life1(
            b1_0, (neighbor_vectors) {
                .a = 0, .b = 0, .c = 0,
                .d=b0_0,        .e = b2_0,
                .f=b0_1.s0,.g=b1_1.s0,.h = b2_1.s0
            }
        );
        uchar8 n2_0 = life1(
            b2_0, (neighbor_vectors) {
                .a = 0, .b = 0, .c = 0,
                .d=b1_0,        .e = 0,
                .f=b1_1.s0,.g=b2_1.s0,.h = 0
            }
        );
        uchar8 n0_1 = life1(
            b0_1, (neighbor_vectors) {
                .a = 0, .b=b0_0.s7,.c=b1_0.s7,
                .d = 0,         .e=b1_1,
                .f=0,   .g=b0_2.s0,.h=b1_2.s0
            }
        );
        uchar8 n1_1 = life1(
            b1_1, (neighbor_vectors) {
                .a=b0_0.s7,.b=b1_0.s7,.c=b2_0.s7,
                .d=b0_1,        .e=b2_1,
                .f=b0_2.s0,.g=b1_2.s0,.h=b2_2.s0
            }
        );
        uchar8 n2_1 = life1(
            b2_1, (neighbor_vectors) {
                .a=b1_0.s7,.b=b2_0.s7,.c=0,
                .d=b1_1,        .e=0,
                .f=b1_2.s0,.g=b2_2.s0,.h=0
            }
        );
        uchar8 n0_2 = life1(
            b0_2, (neighbor_vectors) {
                .a = 0,.b=b0_1.s7,.c=b1_1.s7,
                .d = 0,        .e=b1_2,
                .f = 0,.g = 0, .h=0
            }
        );
        uchar8 n1_2 = life1(
            b1_2, (neighbor_vectors) {
                .a=b0_1.s7,.b=b1_1.s7,.c=b2_1.s7,
                .d=b0_2,        .e=b2_2,
                .f = 0, .g = 0, .h = 0
            }
        );
        uchar8 n2_2 = life1(
            b2_2, (neighbor_vectors) {
                .a=b1_1.s7,.b=b2_1.s7,.c = 0,
                .d=b1_2,        .e = 0,
                .f = 0, .g = 0, .h = 0
            }
        );

        b0_0 = n0_0;
        b1_0 = n1_0;
        b2_0 = n2_0;
        b1_1 = n1_1;
        b2_1 = n2_1;
        b0_2 = n0_2;
        b1_2 = n1_2;
        b2_2 = n2_2;
    }

    // b1_1 now contains the original region advanced by gens_to_simulate generations.
    out[id] = (
        ((ulong)b1_1.s0 << 0)
        | ((ulong)b1_1.s1 << 8)
        | ((ulong)b1_1.s2 << 16)
        | ((ulong)b1_1.s3 << 24)
        | ((ulong)b1_1.s4 << 32)
        | ((ulong)b1_1.s5 << 40)
        | ((ulong)b1_1.s6 << 48)
        | ((ulong)b1_1.s7 << 56)
    );
}
