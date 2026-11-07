#include <cub/cub.cuh>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

struct Param {
  double h, h2, hinv, Lx, Ly, Lz, Lxi, Lyi, Lzi, eta, zeta, kappa, kB, dt, cw, cf, ca, cb, alpha;
  int ncx, ncy, ncz, n;
};
static __constant__ Param c;

#ifndef LUCY
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
__device__ unsigned fkey(int x, int y, int z) {
  x = (x + 2 * c.ncx) % (2 * c.ncx), y = (y + 2 * c.ncy) % (2 * c.ncy), z = (z + 2 * c.ncz) % (2 * c.ncz);
  return cellid(x >> 1, y >> 1, z >> 1) * 8 + (x & 1) + 2 * (y & 1) + 4 * (z & 1);
}
__device__ unsigned cellof(double4 p) {
  return fkey((int)(p.x / c.Lx * 2 * c.ncx) % (2 * c.ncx), (int)(p.y / c.Ly * 2 * c.ncy) % (2 * c.ncy),
              (int)(p.z / c.Lz * 2 * c.ncz) % (2 * c.ncz));
}
__device__ double dist2(double4 a, double4 b, double *dx, double *dy, double *dz) {
  *dx = image(a.x - b.x, c.Lx, c.Lxi), *dy = image(a.y - b.y, c.Ly, c.Lyi),
  *dz = image(a.z - b.z, c.Lz, c.Lzi);
  return fma(*dx, *dx, fma(*dy, *dy, *dz * *dz));
}

__device__ double4 ldcg(const double4 *a) {
  double2 l = __ldcg((const double2 *)a), u = __ldcg((const double2 *)a + 1);
  return make_double4(l.x, l.y, u.x, u.y);
}

#define NEIGHBORS(...)                                                         \
  {                                                                            \
    unsigned kf = key[i];                                                      \
    int k = kf / 8, sb = kf % 8;                                               \
    int fx = 2 * (k % c.ncx) + (sb & 1), fy = 2 * (k / c.ncx % c.ncy) + (sb >> 1 & 1), \
        fz = 2 * (k / (c.ncx * c.ncy)) + (sb >> 2);                          \
    for (int oz = -2; oz <= 2; oz++) {                                         \
      int z = fz + oz;                                                         \
      double sz = z < 0 ? c.Lz : z >= 2 * c.ncz ? -c.Lz : 0;                   \
      for (int oy = -2; oy <= 2; oy++) {                                       \
        int y = fy + oy;                                                       \
        double sy = y < 0 ? c.Ly : y >= 2 * c.ncy ? -c.Ly : 0;                 \
        for (int ox = -2; ox <= 2; ox++) {                                     \
          int x = fx + ox;                                                     \
          double sx = x < 0 ? c.Lx : x >= 2 * c.ncx ? -c.Lx : 0;               \
          unsigned fc = fkey(x, y, z);                                         \
          int cc = fc / 8, e = end[fc];                                        \
          for (int j = start[fc]; j < e; j++) {                                \
            double4 pj = p[j];                                                 \
            double dx = pi.x - pj.x + sx, dy = pi.y - pj.y + sy,               \
                   dz = pi.z - pj.z + sz, r2 = fma(dx, dx, fma(dy, dy, dz * dz)); \
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
    vi.x = fma(ai.x, 0.5 * c.dt, vi.x), vi.y = fma(ai.y, 0.5 * c.dt, vi.y),
    vi.z = fma(ai.z, 0.5 * c.dt, vi.z);
    pi.x = wrap(fma(vi.x, c.dt, pi.x), c.Lx);
    pi.y = wrap(fma(vi.y, c.dt, pi.y), c.Ly);
    pi.z = wrap(fma(vi.z, c.dt, pi.z), c.Lz);
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

__global__ void ranges(const unsigned *key, int shift, int *start, int *end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  unsigned k = key[i] >> shift;
  if (i == 0 || key[i - 1] >> shift != k)
    start[k] = i;
  if (i == c.n - 1 || key[i + 1] >> shift != k)
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
      full[(size_t)m * c.n + i] = j;
    m++;
    if (cc != (int)k && owns(i, j)) {
      if (o < capo)
        own[(size_t)o * c.n + i] = j;
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

__global__ void entropy(double *S, const double *d, const double4 *p, double amp, double kw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < c.n)
    S[i] = log(d[i]) + amp * sin(kw * p[i].y);
}

__global__ void force(const double4 *p, double4 *a, const int *full, const int *nfull) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 pi = p[i];
  double fx = 0, fy = 0, fz = 0;
  int m = nfull[i];
  for (int k = 0; k < m; k++) {
    double4 pj = p[full[(size_t)k * c.n + i]];
    double dx, dy, dz, r2 = dist2(pi, pj, &dx, &dy, &dz);
    double f = (pi.w + pj.w) * F(sqrt(r2));
    fx = fma(f, dx, fx), fy = fma(f, dy, fy), fz = fma(f, dz, fz);
  }
  a[i] = make_double4(fx, fy, fz, 0);
}

__global__ void kick(double4 *v, const double4 *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 vi = v[i], ai = a[i];
  vi.x = fma(ai.x, 0.5 * c.dt, vi.x), vi.y = fma(ai.y, 0.5 * c.dt, vi.y),
  vi.z = fma(ai.z, 0.5 * c.dt, vi.z);
  v[i] = vi;
}

__device__ double uniform(uint64_t key) { return ((mix(key) >> 11) + 0.5) * 0x1p-53; }

__device__ double ln1(double x) {
  double e = x - 1;
  if (fabs(e) > 1e-3)
    return log(x);
  return e * fma(e, fma(e, fma(e, fma(e, 0.2, -0.25), 1.0 / 3), -0.5), 1.0);
}

__device__ double ex1(double y) {
  if (fabs(y) > 1e-3)
    return exp(y);
  return fma(y, fma(y, fma(y, fma(y, fma(y, 1.0 / 120, 1.0 / 24), 1.0 / 6), 0.5), 1.0), 1.0);
}

__device__ double rs1(double x) {
  double e = x - 1;
  if (fabs(e) > 1e-3)
    return rsqrt(x);
  return fma(e, fma(e, fma(e, fma(e, fma(e, -63.0 / 256, 35.0 / 128), -5.0 / 16), 3.0 / 8), -0.5), 1.0);
}

__device__ void pair(int i, int j, uint64_t seed, double tau, const double4 *p, double4 *v,
                     const double *d, const int *id) {
  double dx, dy, dz, r2 = dist2(p[i], p[j], &dx, &dy, &dz);
  if (r2 >= c.h2)
    return;
  double4 vi = ldcg(&v[i]), vj = ldcg(&v[j]);
  double ri = rsqrt(r2), r = r2 * ri, Ti = vi.w, Tj = vj.w;
  double f = F(r) * d[i] * d[j], ex = dx * ri, ey = dy * ri, ez = dz * ri;
  double a = c.ca * f, b = c.cb * f;
  double ux = vi.x - vj.x, uy = vi.y - vj.y, uz = vi.z - vj.z;
  double up = fma(ex, ux, fma(ey, uy, ez * uz));
  double tx = fma(-up, ex, ux), ty = fma(-up, ey, uy), tz = fma(-up, ez, uz);
  uint64_t key = mix(seed ^ (uint64_t)id[i]) ^ (uint64_t)id[j] << 32;
  double si = 1 / (Ti + Tj), tt = Ti * Tj * si, dij = 4 * c.kB * tt * si / 3;
  double var = 4 * c.kB * tt / (1 - dij), Lp = 2 * (4 * a / 3 + b) * tau, Lt = 2 * a * tau;
  double ep = exp(-(1 - dij) * Lp), et = exp(-(1 - dij) * Lt);
  double sp = sqrt(var * (1 - ep * ep)), st = sqrt(var * (1 - et * et));
  float2 g0 = gaussf(key), g1 = gaussf(key + 1), g2 = gaussf(key + 2);
  double gx = g0.x, gy = g0.y, gz = g1.x, gp = g1.y;
  double ge = fma(gx, ex, fma(gy, ey, gz * ez));
  double up1 = fma(up, ep, sp * gp);
  double tx1 = fma(tx, et, st * fma(-ge, ex, gx)), ty1 = fma(ty, et, st * fma(-ge, ey, gy)),
         tz1 = fma(tz, et, st * fma(-ge, ez, gz));
  double wx = 0.5 * (fma(up1, ex, tx1) - ux), wy = 0.5 * (fma(up1, ey, ty1) - uy),
         wz = 0.5 * (fma(up1, ez, tz1) - uz);
  double dK = fma(wx, ux + wx, fma(wy, uy + wy, wz * (uz + wz)));
  double Ti1 = fma(-1.0 / 3, dK, Ti), Tj1 = fma(-1.0 / 3, dK, Tj);
  bool ok = Ti1 > 0 && Tj1 > 0;
#ifndef SSA
  if (ok && sp > 0 && st > 0) {
    double si1 = 1 / (Ti1 + Tj1), tt1 = Ti1 * Tj1 * si1, dij1 = 4 * c.kB * tt1 * si1 / 3;
    double var1 = 4 * c.kB * tt1 / (1 - dij1), dd = dij1 - dij;
    double ep1 = ep * ex1(dd * Lp), et1 = et * ex1(dd * Lt), isp = 1 / sp, ist = 1 / st;
    double rhop = var1 * (1 - ep1 * ep1) * isp * isp, rhot = var1 * (1 - et1 * et1) * ist * ist;
    double isp1 = isp * rs1(rhop), ist1 = ist * rs1(rhot);
    double rp = fma(-ep1, up1, up) * isp1, rx = fma(-et1, tx1, tx) * ist1,
           ry = fma(-et1, ty1, ty) * ist1, rz = fma(-et1, tz1, tz) * ist1;
    double fwd = fma(gp, gp, fma(gx, gx, fma(gy, gy, fma(gz, gz, -ge * ge))));
    double la = c.alpha * ln1(Ti1 * Tj1 / (Ti * Tj)) -
                0.5 * fma(rp, rp, fma(rx, rx, fma(ry, ry, rz * rz))) + 0.5 * fwd -
                0.5 * ln1(rhop) - ln1(rhot);
    ok = la >= 0 || uniform(key + 3) < ex1(la);
  }
#endif
  if (ok) {
    vi.x += wx, vi.y += wy, vi.z += wz, vj.x -= wx, vj.y -= wy, vj.z -= wz;
    Ti = Ti1, Tj = Tj1;
  }
  if (c.kappa > 0) {
    double s = sqrt(4 * c.kappa * c.kB * Ti * Tj * f * tau);
    double q = fma(-2 * c.kappa * f * tau, Ti - Tj, s * g2.x);
    Ti1 = fma(2.0 / 3, q, Ti), Tj1 = fma(-2.0 / 3, q, Tj);
    ok = Ti1 > 0 && Tj1 > 0;
#ifndef SSA
    if (ok && s > 0) {
      double R = Ti1 * Tj1 / (Ti * Tj);
      double back = fma(2 * c.kappa * f * tau, Ti1 - Tj1, -q) / s * rs1(R);
      double la = (c.alpha - 0.5) * ln1(R) - 0.5 * back * back + 0.5 * g2.x * g2.x;
      ok = la >= 0 || uniform(key + 4) < ex1(la);
    }
#endif
    if (ok)
      Ti = Ti1, Tj = Tj1;
  }
  vi.w = Ti, vj.w = Tj;
  v[i] = vi, v[j] = vj;
}

__device__ void within(int s0, int m, int rot, int lane, uint64_t seed, double tau,
                       const double4 *p, double4 *v, const double *d, const int *id) {
  int M = m + (m & 1);
  for (int r0 = 0; r0 < M - 1; r0++) {
    int r = (r0 + rot) % (M - 1);
    for (int q = lane; q < M / 2; q += 32) {
      int a = q == 0 ? M - 1 : (r + q) % (M - 1), b = q == 0 ? r : (r - q + M - 1) % (M - 1);
      if (a < m && b < m)
        pair(s0 + a, s0 + b, seed, tau, p, v, d, id);
    }
    __syncwarp();
  }
}

__device__ void cross(int s0, int m, unsigned rot, int lane, uint64_t seed, double tau,
                      const double4 *p, double4 *v, const double *d, const int *id, const int *own,
                      const int *rem) {
  for (int base = 0; base < m; base += 32) {
    int l = base + lane, i = s0 + l, nj = l < m ? rem[i] : 0, k = 0, j = -1;
    int o = nj ? (nj * lane / 32 + rot) % nj : 0;
    while (__any_sync(0xffffffff, k < nj || j >= 0)) {
      if (j < 0 && k < nj) {
        int q = o + k++;
        j = own[(size_t)(q < nj ? q : q - nj) * c.n + i];
      }
      unsigned same = __match_any_sync(0xffffffff, j);
      if (j >= 0 && __ffs(same) - 1 == lane) {
        pair(i, j, seed, tau, p, v, d, id);
        j = -1;
      }
      __syncwarp();
    }
  }
}

__global__ void dissipate(int color, uint64_t seed, double tau, const double4 *p, double4 *v,
                          const double *d, const int *id, const int *start, const int *end,
                          const int *own, const int *rem) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nx = c.ncx / 3, ny = c.ncy / 3, nz = c.ncz / 3;
  if (w >= nx * ny * nz)
    return;
  int cx = color % 3 + 3 * (w % nx), cy = color / 3 % 3 + 3 * (w / nx % ny),
      cz = color / 9 + 3 * (w / (nx * ny));
  int c0 = cellid(cx, cy, cz), s0 = start[c0], m = end[c0] - s0;
  uint64_t hsh = mix(seed ^ 0x5851f42d4c957f2dULL ^ (uint64_t)c0);
  int rot = m > 2 ? hsh % (m + (m & 1) - 1) : 0;
  if (hsh >> 63) {
    within(s0, m, rot, lane, seed, tau, p, v, d, id);
    cross(s0, m, (unsigned)(hsh >> 32), lane, seed, tau, p, v, d, id, own, rem);
  } else {
    cross(s0, m, (unsigned)(hsh >> 32), lane, seed, tau, p, v, d, id, own, rem);
    within(s0, m, rot, lane, seed, tau, p, v, d, id);
  }
}

__global__ void pressure(double4 *p, const double4 *v, const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < c.n)
    p[i].w = v[i].w * d[i];
}

__global__ void update(double *S, const double4 *v, const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < c.n)
    S[i] = 1.5 * log(v[i].w * cbrt(d[i] * d[i]));
}

#define NQ 9

__global__ void diag(int mode, double kw, const double4 *p, const double4 *v, const double *S,
                     double *val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.n)
    return;
  double4 pi = p[i], vi = v[i];
  double v2 = fma(vi.x, vi.x, fma(vi.y, vi.y, vi.z * vi.z));
  double q[NQ] = {(mode == 2 ? S[i] : mode ? vi.y : vi.x) * sin(kw * pi.y), fma(1.5, vi.w, 0.5 * v2),
                  S[i], v2, vi.w, vi.w * vi.w, vi.x, vi.y, vi.z};
  for (int k = 0; k < NQ; k++)
    val[(size_t)k * c.n + i] = q[k];
}

static int n, nb, ncells, bits, *fstart, *fend;

#define cudaMalloc(p, size) alloc((void **)(p), (size_t)(size))
static void alloc(void **p, size_t size) {
  if ((cudaMalloc)(p, size) != cudaSuccess) {
    fprintf(stderr, "sdpd3cu: cannot allocate %zu bytes\n", size);
    exit(1);
  }
}
static double4 *p, *v, *a, *p2, *v2;
static double *S, *S2, *d, *val, *sum;
static int *id, *id2, *idx, *idx2, *start, *end, *full, *nfull, *own, *rem, *over, cap = 160,
    capo = 96, nbw;
static unsigned *key, *key2;
static void *tmp, *rtmp;
static size_t ntmp, nrtmp;
static int *seg;

static void sort(int move) {
  drift<<<nb, 256>>>(move, p, v, a, key, idx);
  cub::DeviceRadixSort::SortPairs(tmp, ntmp, key, key2, idx, idx2, n, 0, bits);
  gather<<<nb, 256>>>(idx2, p, v, S, id, p2, v2, S2, id2);
  std::swap(p, p2), std::swap(v, v2), std::swap(S, S2), std::swap(id, id2);
  cudaMemsetAsync(start, 0, ncells * sizeof *start);
  cudaMemsetAsync(end, 0, ncells * sizeof *end);
  cudaMemsetAsync(fstart, 0, 8 * ncells * sizeof *fstart);
  cudaMemsetAsync(fend, 0, 8 * ncells * sizeof *fend);
  ranges<<<nb, 256>>>(key2, 3, start, end);
  ranges<<<nb, 256>>>(key2, 0, fstart, fend);
}

static void neighbors(void) {
  int o[2];
  for (;;) {
    cudaMemsetAsync(over, 0, 2 * sizeof *over);
    build<<<nb, 256>>>(p, v, S, d, key2, fstart, fend, cap, full, nfull, capo, own, rem, over);
    cudaMemcpy(o, over, sizeof o, cudaMemcpyDeviceToHost);
    if (o[0] == 0 && o[1] == 0)
      break;
    if (o[0])
      cudaFree(full), cap = o[0] + 16, cudaMalloc(&full, (size_t)cap * n * sizeof *full);
    if (o[1])
      cudaFree(own), capo = o[1] + 16, cudaMalloc(&own, (size_t)capo * n * sizeof *own);
  }
}

static void permute(uint64_t s, int *perm) {
  for (int k = 0; k < 27; k++)
    perm[k] = k;
  for (int k = 26; k > 0; k--) {
    int j = (s = mix(s + 0x9e3779b97f4a7c15ULL)) % (k + 1);
    std::swap(perm[k], perm[j]);
  }
}

static void relax(uint64_t tick, double tau) {
  int perm[27];
  permute(mix(tick ^ 0xd1342543de82ef95ULL), perm);
  for (int k = 0; k < 27; k++)
    dissipate<<<nbw, 128>>>(perm[k], mix(tick), tau, p, v, d, id, start, end, own, rem);
  update<<<nb, 256>>>(S, v, d);
  pressure<<<nb, 256>>>(p, v, d);
  force<<<nb, 256>>>(p, a, full, nfull);
}

int main(int argc, char **argv) {
  int mode, rows, waves, ncell, nyc, i, j, k, l, b, step, steps, every, dump, owed;
  uint64_t tick;
  double A, lambda, kw, amp, h;
  double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  Param hc;
  cudaEvent_t t0, t1;
  float ms;
  if (argc != 11 && argc != 13) {
    fprintf(stderr,
            "usage: %s shear|sound|heat|rest h rows eta zeta kappa kB dt steps every "
            "[amp dumpevery]\n",
            argv[0]);
    return 1;
  }
  mode = argv[1][0] == 'h' ? 2 : argv[1][1] == 'o';
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
  hc.alpha = hc.kB > 0 ? 1.5 / hc.kB - 1 : 0;
  hc.Lx = hc.Lz = ncell * A;
  hc.Ly = nyc * A;
  hc.Lxi = 1 / hc.Lx, hc.Lyi = 1 / hc.Ly, hc.Lzi = 1 / hc.Lz;
  n = hc.n = 4 * ncell * ncell * nyc;
  hc.ncx = hc.ncz = 3 * (int)(hc.Lx / h / 3);
  hc.ncy = 3 * (int)(hc.Ly / h / 3);
#ifndef LUCY
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
  for (bits = 1; (1 << bits) < 8 * ncells; bits++)
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
          hv[l] = make_double4(mode == 0 ? amp * sin(kw * y) : 0, mode == 1 ? amp * sin(kw * y) : 0,
                               0, 0);
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
  cudaMalloc(&fstart, 8 * ncells * sizeof *fstart), cudaMalloc(&fend, 8 * ncells * sizeof *fend);
  cudaMalloc(&val, (size_t)NQ * n * sizeof *val), cudaMalloc(&sum, NQ * sizeof *sum);
  {
    int hseg[NQ + 1];
    for (k = 0; k <= NQ; k++)
      hseg[k] = k * n;
    cudaMalloc(&seg, sizeof hseg);
    cudaMemcpy(seg, hseg, sizeof hseg, cudaMemcpyHostToDevice);
    cub::DeviceSegmentedReduce::Sum(rtmp, nrtmp, val, sum, NQ, seg, seg + 1);
    cudaMalloc(&rtmp, nrtmp);
  }
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
  entropy<<<nb, 256>>>(S, d, p, mode == 2 ? amp : 0, kw);
  neighbors();
  force<<<nb, 256>>>(p, a, full, nfull);

  printf("# lambda/h %.6g k %.10g n %d\n", lambda / h, kw, n);
  cudaEventCreate(&t0), cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (step = 0, tick = 0, owed = 0; step <= steps; step++) {
    if (owed && ((dump > 0 && step % dump == 0) || step % every == 0))
      relax(++tick, hc.dt / 2), owed = 0;
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
      double s[NQ];
      diag<<<nb, 256>>>(mode, kw, p, v, S, val);
      cub::DeviceSegmentedReduce::Sum(rtmp, nrtmp, val, sum, NQ, seg, seg + 1);
      cudaMemcpy(s, sum, sizeof s, cudaMemcpyDeviceToHost);
      printf("%.10g %.10g %.15g %.15g %.10g %.10g %.10g %.3g %.3g %.3g\n", step * hc.dt,
             2 * s[0] / n, s[1], s[2], s[3] / (3 * n), hc.kB * s[4] / n,
             s[5] * n / (s[4] * s[4]) - 1, s[6], s[7], s[8]);
    }
    if (step == steps)
      break;
    relax(++tick, owed ? hc.dt : hc.dt / 2);
    sort(1);
    neighbors();
    force<<<nb, 256>>>(p, a, full, nfull);
    kick<<<nb, 256>>>(v, a);
    owed = 1;
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
