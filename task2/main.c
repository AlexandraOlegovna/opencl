#define SWAP(a, b) {__local float * tmp=a; a=b; b=tmp;}

__kernel void scan(
                __global float* a,
                __global float* r,
                __local float* b,
                __local float* c) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    c[lid] = b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s <<= 1) {
        if (lid > (s - 1)) {
            c[lid] = b[lid] + b[lid - s];
        } else {
            c[lid] = b[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(b, c);
    }
    r[gid] = b[lid];
}

__kernel void combine(
        __global float* a,
        __global float* g,
        __global float* b) {
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    uint block_size = get_local_size(0);
    b[lid + gid * block_size] = a[lid + gid * block_size] + g[gid - 1];
}