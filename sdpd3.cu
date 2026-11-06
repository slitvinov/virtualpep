#include <cub/cub.cuh>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

struct Param {
  double h, h2, hinv, Lx, Ly, Lz, Lxi, Lyi, Lzi, eta, zeta, kappa, kB, dt, cw, cf, ca, cb;
  int ncx, ncy, ncz, n;
};
static __constant__ Param c;

#ifdef WENDLAND
__device__ double W(double r) {
  double q = r * c.hinv, s = 1 - q;
  return c.cw * (1 + 4 * q) * s * s * s * s;
}

__device__ double F(double r) {
  double s = 1 - r * c.hinv;
  return c.cf * s * s * s;
}
#else
__device__ double W(double r) {
  double q = r * c.hinv, s = 1 - q;
  return c.cw * (1 + 3 * q) * s * s * s;
}

__device__ double F(double r) {
  double s = 1 - r * c.hinv;
  return c.cf * s * s;
}
#endif

__host__ __device__ uint64_t mix(uint64_t w) {
  w = (w ^ (w >> 30)) * 0xbf58476d1ce4e5b9ULL;
  w = (w ^ (w >> 27)) * 0x94d049bb133111ebULL;
  return w ^ (w >> 31);
}

__device__ float2 gaussf(uint64_t key) {
  uint64_t a = mix(key);
  float u = ((unsigned)a >> 8) * 0x1p-24f + 0x1p-25f, w = (unsigned)(a >> 40) * 0x1p-24f, sn, cs;
  sincospif(2 * w, &sn, &cs);
  float m = sqrtf(-2 * __logf(u));
  return make_float2(m * cs, m * sn);
}

__device__ double wrap(double a, double L) {
  a -= L * floor(a / L);
  return a < L ? a : 0;
}
__device__ double image(double a, double L, double Li) { return a - L * round(a * Li); }
__device__ int cellid(int cx, int cy, int cz) {
  return (cx + c.ncx) % c.ncx + c.ncx * ((cy + c.ncy) % c.ncy + c.ncy * ((cz + c.ncz) % c.ncz));
}
__device__ unsigned cellof(double4 p) {
  return cellid((int)(p.x / c.Lx * c.ncx) % c.ncx, (int)(p.y / c.Ly * c.ncy) % c.ncy,
                (int)(p.z / c.Lz * c.ncz) % c.ncz);
}
__device__ double dist2(double4 a, double4 b, double *dx, double *dy, double *dz) {
  *dx = image(a.x - b.x, c.Lx, c.Lxi), *dy = image(a.y - b.y, c.Ly, c.Lyi),
  *dz = image(a.z - b.z, c.Lz, c.Lzi);
  return *dx * *dx + *dy * *dy + *dz * *dz;
}

__device__ double4 ldcg(const double4 *a) {
  double2 l = __ldcg((const double2 *)a), u = __ldcg((const double2 *)a + 1);
  return make_double4(l.x, l.y, u.x, u.y);
}

#define NEIGHBORS(...)                                                         \
  {                                                                            \
    unsigned k = key[i];                                                       \
    int cx = k % c.ncx, cy = k / c.ncx % c.ncy, cz = k / (c.ncx * c.ncy);      \
    for (int oz = -1; oz <= 1; oz++) {                                         \
      int z = cz + oz;                                                         \
      double sz = z < 0 ? c.Lz : z >= c.ncz ? -c.Lz : 0;                       \
      for (int oy = -1; oy <= 1; oy++) {                                       \
        int y = cy + oy;                                                       \
        double sy = y < 0 ? c.Ly : y >= c.ncy ? -c.Ly : 0;                     \
        for (int ox = -1; ox <= 1; ox++) {                                     \
          int x = cx + ox;                                                     \
          double sx = x < 0 ? c.Lx : x >= c.ncx ? -c.Lx : 0;                   \
          int cc = cellid(x, y, z), e = end[cc];                               \
          for (int j = start[cc]; j < e; j++) {                                \
            double4 pj = p[j];                                                 \
            double dx = pi.x - pj.x + sx, dy = pi.y - pj.y + sy,               \
                   dz = pi.z - pj.z + sz, r2 = dx * dx + dy * dy + dz * dz;    \
            if (r2 < c.h2) {                                                   \
              __VA_ARGS__                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

__global__ void drift(int move, double4 *p, double4 *v, const double4 *a, unsigned *key,
                      int *idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 pi = p[i];
  if (move) {
    double4 vi = v[i], ai = a[i];
    vi.x += ai.x * c.dt / 2, vi.y += ai.y * c.dt / 2, vi.z += ai.z * c.dt / 2;
    pi.x = wrap(pi.x + vi.x * c.dt, c.Lx);
    pi.y = wrap(pi.y + vi.y * c.dt, c.Ly);
    pi.z = wrap(pi.z + vi.z * c.dt, c.Lz);
    v[i] = vi, p[i] = pi;
  }
  key[i] = cellof(pi);
  idx[i] = i;
}

__global__ void gather(const int *idx, const double4 *p, const double4 *v, const double *S,
                       const int *id, double4 *p2, double4 *v2, double *S2, int *id2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  int k = idx[i];
  p2[i] = p[k], v2[i] = v[k], S2[i] = S[k], id2[i] = id[k];
}

__global__ void ranges(const unsigned *key, int *start, int *end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  unsigned k = key[i];
  if (i == 0 || key[i - 1] != k)
    start[k] = i;
  if (i == c.n - 1 || key[i + 1] != k)
    end[k] = i + 1;
}

__device__ bool owns(int i, int j) { return (i + j) & 1 ? i < j : i > j; }

__global__ void build(double4 *p, double4 *v, const double *S, double *d, const unsigned *key,
                      const int *start, const int *end, int cap, int *full, int *nfull, int capo,
                      int *own, int *rem, int *over) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 pi = p[i];
  double s = 0;
  int m = 0, o = 0;
  NEIGHBORS(s += W(sqrt(r2)); if (j != i) {
    if (m < cap)
      full[m * c.n + i] = j;
    m++;
    if (cc != (int)k && owns(i, j)) {
      if (o < capo)
        own[o * c.n + i] = j;
      o++;
    }
  })
  if (m > cap)
    atomicMax(&over[0], m);
  if (o > capo)
    atomicMax(&over[1], o);
  double T = cbrt(s * s) * exp(2 * S[i] / 3);
  nfull[i] = m, rem[i] = o, d[i] = 1 / s;
  v[i].w = T;
  p[i].w = T / s;
}

__global__ void entropy(double *S, const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < c.n)
    S[i] = log(d[i]);
}

__global__ void force(const double4 *p, double4 *a, const int *full, const int *nfull) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 pi = p[i];
  double fx = 0, fy = 0, fz = 0;
  int m = nfull[i];
  for (int k = 0; k < m; k++) {
    double4 pj = p[full[k * c.n + i]];
    double dx, dy, dz, r2 = dist2(pi, pj, &dx, &dy, &dz);
    double f = (pi.w + pj.w) * F(sqrt(r2));
    fx += f * dx, fy += f * dy, fz += f * dz;
  }
  a[i] = make_double4(fx, fy, fz, 0);
}

__global__ void kick(double4 *v, const double4 *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 vi = v[i], ai = a[i];
  vi.x += ai.x * c.dt / 2, vi.y += ai.y * c.dt / 2, vi.z += ai.z * c.dt / 2;
  v[i] = vi;
}

__device__ void pair(int i, int j, uint64_t seed, const double4 *p, double4 *v, const double *d,
                     const int *id) {
  double dx, dy, dz, r2 = dist2(p[i], p[j], &dx, &dy, &dz);
  if (r2 >= c.h2)
    return;
  double4 vi = ldcg(&v[i]), vj = ldcg(&v[j]);
  double ri = rsqrt(r2), r = r2 * ri, Ti = vi.w, Tj = vj.w;
  double f = F(r) * d[i] * d[j], ex = dx * ri, ey = dy * ri, ez = dz * ri;
  double a = c.ca * f, b = c.cb * f;
  double si = 1 / (Ti + Tj), tt = Ti * Tj * si, dij = 4 * c.kB * tt * si / 3;
  double ux = vi.x - vj.x, uy = vi.y - vj.y, uz = vi.z - vj.z;
  double up = ex * ux + ey * uy + ez * uz;
  double tx = ux - up * ex, ty = uy - up * ey, tz = uz - up * ez;
  uint64_t key = mix(seed ^ (uint64_t)id[i]) ^ (uint64_t)id[j] << 32;
  double ep = exp(-2 * (1 - dij) * (4 * a / 3 + b) * c.dt), et = exp(-2 * (1 - dij) * a * c.dt);
  double var = 4 * c.kB * tt / (1 - dij);
  double sp = sqrt(var * (1 - ep * ep)), st = sqrt(var * (1 - et * et));
  float2 g0 = gaussf(key), g1 = gaussf(key + 1), g2 = gaussf(key + 2);
  double gx = g0.x, gy = g0.y, gz = g1.x;
  double ge = gx * ex + gy * ey + gz * ez;
  double up1 = up * ep + sp * g1.y;
  double tx1 = tx * et + st * (gx - ge * ex), ty1 = ty * et + st * (gy - ge * ey),
         tz1 = tz * et + st * (gz - ge * ez);
  double wx = 0.5 * (up1 * ex + tx1 - ux), wy = 0.5 * (up1 * ey + ty1 - uy),
         wz = 0.5 * (up1 * ez + tz1 - uz);
  double dK = 0.25 * (up1 * up1 + tx1 * tx1 + ty1 * ty1 + tz1 * tz1 - ux * ux - uy * uy - uz * uz);
  double q = -2 * c.kappa * f * (Ti - Tj) * c.dt + sqrt(4 * c.kappa * c.kB * Ti * Tj * f * c.dt) * g2.x;
  vi.x += wx, vi.y += wy, vi.z += wz, vi.w += (q - 0.5 * dK) * (2.0 / 3);
  vj.x -= wx, vj.y -= wy, vj.z -= wz, vj.w += (-q - 0.5 * dK) * (2.0 / 3);
  v[i] = vi, v[j] = vj;
}

__global__ void dissipate(int color, uint64_t seed, const double4 *p, double4 *v, const double *d,
                          const int *id, const int *start, const int *end, const int *own,
                          const int *rem) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nx = c.ncx / 3, ny = c.ncy / 3, nz = c.ncz / 3;
  if (w >= nx * ny * nz)
    return;
  int cx = color % 3 + 3 * (w % nx), cy = color / 3 % 3 + 3 * (w / nx % ny),
      cz = color / 9 + 3 * (w / (nx * ny));
  int c0 = cellid(cx, cy, cz), s0 = start[c0], m = end[c0] - s0, M = m + (m & 1);
  for (int r = 0; r < M - 1; r++) {
    for (int q = lane; q < M / 2; q += 32) {
      int a = q == 0 ? M - 1 : (r + q) % (M - 1), b = q == 0 ? r : (r - q + M - 1) % (M - 1);
      if (a < m && b < m)
        pair(s0 + a, s0 + b, seed, p, v, d, id);
    }
    __syncwarp();
  }
  for (int base = 0; base < m; base += 32) {
    int l = base + lane, i = s0 + l, nj = l < m ? rem[i] : 0, o = nj * lane / 32, k = 0, j = -1;
    while (__any_sync(0xffffffff, k < nj || j >= 0)) {
      if (j < 0 && k < nj) {
        int q = o + k++;
        j = own[(q < nj ? q : q - nj) * c.n + i];
      }
      unsigned same = __match_any_sync(0xffffffff, j);
      if (j >= 0 && __ffs(same) - 1 == lane) {
        pair(i, j, seed, p, v, d, id);
        j = -1;
      }
      __syncwarp();
    }
  }
}

__global__ void update(double *S, const double4 *v, const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < c.n)
    S[i] = 1.5 * log(v[i].w * cbrt(d[i] * d[i]));
}

__global__ void diag(int mode, double kw, const double4 *p, const double4 *v, const double *S,
                     double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s[5] = {0, 0, 0, 0, 0};
  if (i < c.n) {
    double4 pi = p[i], vi = v[i];
    double v2 = vi.x * vi.x + vi.y * vi.y + vi.z * vi.z;
    s[0] = (mode ? vi.y : vi.x) * sin(kw * pi.y);
    s[1] = v2 / 2 + 1.5 * vi.w;
    s[2] = S[i];
    s[3] = v2;
    s[4] = vi.w;
  }
  for (int k = 0; k < 5; k++) {
    for (int o = 16; o > 0; o /= 2)
      s[k] += __shfl_down_sync(0xffffffff, s[k], o);
    if ((threadIdx.x & 31) == 0)
      atomicAdd(&out[k], s[k]);
  }
}

static int n, nb, ncells, bits;
static double4 *p, *v, *a, *p2, *v2;
static double *S, *S2, *d, *out;
static int *id, *id2, *idx, *idx2, *start, *end, *full, *nfull, *own, *rem, *over, cap = 160,
    capo = 96, nbw;
static unsigned *key, *key2;
static void *tmp;
static size_t ntmp;

static void sort(int move) {
  drift<<<nb, 256>>>(move, p, v, a, key, idx);
  cub::DeviceRadixSort::SortPairs(tmp, ntmp, key, key2, idx, idx2, n, 0, bits);
  gather<<<nb, 256>>>(idx2, p, v, S, id, p2, v2, S2, id2);
  std::swap(p, p2), std::swap(v, v2), std::swap(S, S2), std::swap(id, id2);
  cudaMemsetAsync(start, 0, ncells * sizeof *start);
  cudaMemsetAsync(end, 0, ncells * sizeof *end);
  ranges<<<nb, 256>>>(key2, start, end);
}

static void neighbors(void) {
  int o[2];
  for (;;) {
    cudaMemsetAsync(over, 0, 2 * sizeof *over);
    build<<<nb, 256>>>(p, v, S, d, key2, start, end, cap, full, nfull, capo, own, rem, over);
    cudaMemcpy(o, over, sizeof o, cudaMemcpyDeviceToHost);
    if (o[0] == 0 && o[1] == 0)
      break;
    if (o[0])
      cudaFree(full), cap = o[0] + 16, cudaMalloc(&full, (size_t)cap * n * sizeof *full);
    if (o[1])
      cudaFree(own), capo = o[1] + 16, cudaMalloc(&own, (size_t)capo * n * sizeof *own);
  }
}

static void dissipate(uint64_t tick) {
  for (int color = 0; color < 27; color++)
    dissipate<<<nbw, 128>>>(color, mix(tick), p, v, d, id, start, end, own, rem);
}

int main(int argc, char **argv) {
  int mode, rows, waves, ncell, nyc, i, j, k, l, b, step, steps, every, dump;
  double A, lambda, kw, amp, h;
  double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  Param hc;
  cudaEvent_t t0, t1;
  float ms;
  if (argc != 11 && argc != 13) {
    fprintf(stderr,
            "usage: %s shear|sound|rest h rows eta zeta kappa kB dt steps every "
            "[amp dumpevery]\n",
            argv[0]);
    return 1;
  }
  mode = argv[1][1] == 'o';
  h = atof(argv[2]);
  rows = atoi(argv[3]);
  hc.eta = atof(argv[4]);
  hc.zeta = atof(argv[5]);
  hc.kappa = atof(argv[6]);
  hc.kB = atof(argv[7]);
  hc.dt = atof(argv[8]);
  steps = atoi(argv[9]);
  every = atoi(argv[10]);
  dump = argc == 13 ? atoi(argv[12]) : 0;

  A = cbrt(4);
  lambda = rows * A / 2;
  waves = (int)ceil(3 * h / lambda);
  if (rows * waves % 2)
    waves++;
  nyc = rows * waves / 2;
  ncell = (int)ceil((dump != 0 && lambda > 3 * h ? lambda : 3 * h) / A);
  hc.h = h, hc.h2 = h * h, hc.hinv = 1 / h;
  hc.ca = 5 * hc.eta / 3 - hc.zeta, hc.cb = 16 * hc.zeta / 3 + 10 * hc.eta / 9;
  hc.Lx = hc.Lz = ncell * A;
  hc.Ly = nyc * A;
  hc.Lxi = 1 / hc.Lx, hc.Lyi = 1 / hc.Ly, hc.Lzi = 1 / hc.Lz;
  n = hc.n = 4 * ncell * ncell * nyc;
  hc.ncx = hc.ncz = 3 * (int)(hc.Lx / h / 3);
  hc.ncy = 3 * (int)(hc.Ly / h / 3);
#ifdef WENDLAND
  hc.cw = 21 / (2 * M_PI * h * h * h);
  hc.cf = 210 / (M_PI * h * h * h * h * h);
#else
  hc.cw = 105 / (16 * M_PI * h * h * h);
  hc.cf = 315 / (4 * M_PI * h * h * h * h * h);
#endif
  cudaMemcpyToSymbol(c, &hc, sizeof hc);
  kw = 2 * M_PI / lambda;
  amp = argc == 13 ? atof(argv[11]) : argv[1][0] == 'r' ? 0 : 1e-3;
  nb = (n + 255) / 256;
  ncells = hc.ncx * hc.ncy * hc.ncz;
  for (bits = 1; (1 << bits) < ncells; bits++)
    ;

  double4 *hp = (double4 *)malloc(n * sizeof *hp), *hv = (double4 *)malloc(n * sizeof *hv);
  double *hS = (double *)malloc(n * sizeof *hS);
  int *hid = (int *)malloc(n * sizeof *hid);
  for (l = 0, k = 0; k < ncell; k++)
    for (j = 0; j < nyc; j++)
      for (i = 0; i < ncell; i++)
        for (b = 0; b < 4; b++, l++) {
          double y = (j + basis[b][1]) * A;
          hp[l] = make_double4((i + basis[b][0]) * A, y, (k + basis[b][2]) * A, 0);
          hv[l] = make_double4(mode ? 0 : amp * sin(kw * y), mode ? amp * sin(kw * y) : 0, 0, 0);
          hS[l] = 0;
          hid[l] = l;
        }
  cudaMalloc(&p, n * sizeof *p), cudaMalloc(&p2, n * sizeof *p2);
  cudaMalloc(&v, n * sizeof *v), cudaMalloc(&v2, n * sizeof *v2);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&S, n * sizeof *S), cudaMalloc(&S2, n * sizeof *S2), cudaMalloc(&d, n * sizeof *d);
  cudaMalloc(&id, n * sizeof *id), cudaMalloc(&id2, n * sizeof *id2);
  cudaMalloc(&idx, n * sizeof *idx), cudaMalloc(&idx2, n * sizeof *idx2);
  cudaMalloc(&key, n * sizeof *key), cudaMalloc(&key2, n * sizeof *key2);
  cudaMalloc(&start, ncells * sizeof *start), cudaMalloc(&end, ncells * sizeof *end);
  cudaMalloc(&out, 5 * sizeof *out);
  cudaMalloc(&full, (size_t)cap * n * sizeof *full), cudaMalloc(&nfull, n * sizeof *nfull);
  cudaMalloc(&own, (size_t)capo * n * sizeof *own), cudaMalloc(&rem, n * sizeof *rem);
  cudaMalloc(&over, 2 * sizeof *over);
  nbw = (32 * (hc.ncx / 3) * (hc.ncy / 3) * (hc.ncz / 3) + 127) / 128;
  cub::DeviceRadixSort::SortPairs(tmp, ntmp, key, key2, idx, idx2, n, 0, bits);
  cudaMalloc(&tmp, ntmp);
  cudaMemcpy(p, hp, n * sizeof *p, cudaMemcpyHostToDevice);
  cudaMemcpy(v, hv, n * sizeof *v, cudaMemcpyHostToDevice);
  cudaMemcpy(S, hS, n * sizeof *S, cudaMemcpyHostToDevice);
  cudaMemcpy(id, hid, n * sizeof *id, cudaMemcpyHostToDevice);

  sort(0);
  neighbors();
  entropy<<<nb, 256>>>(S, d);
  neighbors();
  force<<<nb, 256>>>(p, a, full, nfull);

  printf("# lambda/h %.6g k %.10g n %d\n", lambda / h, kw, n);
  cudaEventCreate(&t0), cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (step = 0; step <= steps; step++) {
    if (dump > 0 && step % dump == 0) {
      char name[64];
      FILE *file;
      cudaMemcpy(hp, p, n * sizeof *p, cudaMemcpyDeviceToHost);
      cudaMemcpy(hv, v, n * sizeof *v, cudaMemcpyDeviceToHost);
      snprintf(name, sizeof name, "dump.%06d", step / dump);
      file = fopen(name, "w");
      fprintf(file, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n0 %.10g\n0 %.10g\n0 %.10g\nITEM: ATOMS x y z vx vy vz T\n", step, n, hc.Lx, hc.Ly, hc.Lz);
      for (i = 0; i < n; i++)
        fprintf(file, "%.6g %.6g %.6g %.6g %.6g %.6g %.6g\n", hp[i].x, hp[i].y, hp[i].z, hv[i].x,
                hv[i].y, hv[i].z, hv[i].w);
      fclose(file);
    }
    if (step % every == 0) {
      double s[5];
      cudaMemset(out, 0, 5 * sizeof *out);
      diag<<<nb, 256>>>(mode, kw, p, v, S, out);
      cudaMemcpy(s, out, sizeof s, cudaMemcpyDeviceToHost);
      printf("%.10g %.10g %.15g %.15g %.10g %.10g\n", step * hc.dt, 2 * s[0] / n, s[1], s[2],
             s[3] / (3 * n), hc.kB * s[4] / n);
    }
    sort(1);
    neighbors();
    force<<<nb, 256>>>(p, a, full, nfull);
    kick<<<nb, 256>>>(v, a);
    dissipate((uint64_t)step + 1);
    update<<<nb, 256>>>(S, v, d);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  cudaEventElapsedTime(&ms, t0, t1);
  fprintf(stderr, "%s: n %d steps %d time %.3f s %.3g particle-steps/s\n", argv[0], n, steps,
          ms / 1e3, (double)n * (steps + 1) / (ms / 1e3));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", argv[0], cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
