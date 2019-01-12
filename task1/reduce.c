kernel void conv(global float* A, global float* B, global float* output, int N, int M, int HM) {
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= N || col >= N)
      return;

   float res = 0;

   for (int k = 0; k < M; ++k) {
      for (int l = 0; l < M; ++l) {
         int irow = row + k - HM;
         int icol = col + l - HM;
         int input_idx = irow * N + icol;
         float Aval = 0 <= irow && irow < N && 0 <= icol && icol < N ? A[irow * N + icol] : 0;
         res += Aval * B[k * M + l];
      }
   }

   output[row * N + col] = res;
}