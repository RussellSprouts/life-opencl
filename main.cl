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
    ushort16 sum;
    ushort16 carry;
} adder_result;

adder_result half_adder(ushort16 a, ushort16 b) {
    ushort16 sum = a ^ b;
    ushort16 carry = a & b;
    return (adder_result) {
        .sum = sum,
        .carry = carry
    };
}

adder_result full_adder(ushort16 a, ushort16 b, ushort16 c) {
    ushort16 temp = a ^ b;
    ushort16 sum = temp ^ c;
    ushort16 carry = (a & b) | (temp & c);
    return (adder_result) {
        .sum = sum,
        .carry = carry
    };
}

/**
 * @brief Processes one generation of life on a 16x16 torus.
 * 
 * @param grid 
 * @return ushort16 The new grid
 */
ushort16 life16x16(const ushort16 grid) {
    // abc
    // dXe
    // fgh
    ushort16 a = rotate(grid, -1).sf0123456789abcde;
    ushort16 b = grid.sf0123456789abcde;
    ushort16 c = rotate(grid, 1).sf0123456789abcde;

    adder_result li = full_adder(a, b, c);

    ushort16 d = rotate(grid, -1);
    ushort16 e = rotate(grid, 1);

    adder_result mj = half_adder(d, e);

    ushort16 f = rotate(grid, -1).s123456789abcdef0;
    ushort16 g = grid.s123456789abcdef0;
    ushort16 h = rotate(grid, 1).s123456789abcdef0;

    adder_result nk = full_adder(g, h, f);

    adder_result yw = full_adder(li.sum, mj.sum, nk.sum);
    adder_result xz = full_adder(li.carry, mj.carry, nk.carry);

    // survive if currently alive
    ushort16 result = grid;
    // born if 1,3,5,7 neighbors
    result |= yw.sum;
    // survive if 2,3,6,7 neighbors
    result &= (yw.carry ^ xz.sum);
    // survive if 0,1,2,3 neighbors
    result &= ~xz.carry;

    return result;
}

/**
 * @brief Simulates up to 4 generations of Life.
 * regions is an array of 8x8 tiles packed as 64 bit ints.
 * neighbors contains 8 neighbor indices for each region,
 *   pointing to the 8 neighbors of that region.
 * out is the resulting pattern of all the regions after
 *   gens_to_simulate generations.
 * 
 * @param gens_to_simulate 
 * @param regions 
 * @param neighbors 
 * @param out 
 */
kernel void sparse_life4(const int gens_to_simulate, const global ulong *regions, const global ushort *neighbors, global ulong *out) {
    int id = get_global_id(0);
    // read the 24 x 24 region around the region.
    uchar8 b0_0 = block(regions[neighbors[id * 8]]);
    uchar8 b1_0 = block(regions[neighbors[id * 8 + 1]]);
    uchar8 b2_0 = block(regions[neighbors[id * 8 + 2]]);

    uchar8 b0_1 = block(regions[neighbors[id * 8 + 3]]);
    uchar8 b1_1 = block(regions[id]);
    uchar8 b2_1 = block(regions[neighbors[id * 8 + 4]]);

    uchar8 b0_2 = block(regions[neighbors[id * 8 + 5]]);
    uchar8 b1_2 = block(regions[neighbors[id * 8 + 6]]);
    uchar8 b2_2 = block(regions[neighbors[id * 8 + 7]]);

    // Get the central 16x16 region of that.
    ushort8 top = (convert_ushort8(b0_0) << 12) + (convert_ushort8(b1_0) << 4) + convert_ushort8(b2_0);
    ushort8 middle = (convert_ushort8(b0_1) << 12) + (convert_ushort8(b1_1) << 4) + convert_ushort8(b2_1);
    ushort8 bottom = (convert_ushort8(b0_2) << 12) + (convert_ushort8(b1_2) << 4) + convert_ushort8(b2_2);

    ushort16 grid = (ushort16)(
        (ushort8)(top.s4567, middle.s0123),
        (ushort8)(middle.s4567, bottom.s0123)
    );

    // Simulate up to 4 generations
    for (int i = 0; i < gens_to_simulate; i++) {
        grid = life16x16(grid);
    }

    // Output the central 8x8 region.
    uchar8 central = convert_uchar16(grid >> 4).s456789ab;
    out[id] = (
        ((ulong)central.s0 << 0)
        | ((ulong)central.s1 << 8)
        | ((ulong)central.s2 << 16)
        | ((ulong)central.s3 << 24)
        | ((ulong)central.s4 << 32)
        | ((ulong)central.s5 << 40)
        | ((ulong)central.s6 << 48)
        | ((ulong)central.s7 << 56)
    );
}
