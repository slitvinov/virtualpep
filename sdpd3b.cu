#include <cub/cub.cuh>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

#define MAXB 256

struct Param {
  double h, h2, hinv, Lx, Ly, Lz, Lxi, Lyi, Lzi, eta, zeta, kappa, kB, dt, cw, cf, ca, cb, amp, kw;
  uint64_t seed;
  int ncx, ncy, ncz, n, cell, mode;
};
static __constant__ Param cs[MAXB];
static __constant__ int cn, nbox, wb[MAXB + 1];

struct Sched {
  double tau;
  uint64_t seed;
  int perm[27];
};

#ifndef LUCY
__device__ double W(const Param &c, double r) {
  double q = r * c.hinv, s = 1 - q;
  return c.cw * (1 + 4 * q) * s * s * s * s;
}

__device__ double F(const Param &c, double r) {
  double s = 1 - r * c.hinv;
  return c.cf * s * s * s;
}
#else
__device__ double W(const Param &c, double r) {
  double q = r * c.hinv, s = 1 - q;
  return c.cw * (1 + 3 * q) * s * s * s;
}

__device__ double F(const Param &c, double r) {
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
__device__ int cellid(const Param &c, int cx, int cy, int cz) {
  return (cx + c.ncx) % c.ncx + c.ncx * ((cy + c.ncy) % c.ncy + c.ncy * ((cz + c.ncz) % c.ncz));
}
__device__ unsigned cellof(const Param &c, double4 p) {
  return c.cell + cellid(c, (int)(p.x / c.Lx * c.ncx) % c.ncx, (int)(p.y / c.Ly * c.ncy) % c.ncy,
                         (int)(p.z / c.Lz * c.ncz) % c.ncz);
}
__device__ double dist2(const Param &c, double4 a, double4 b, double *dx, double *dy, double *dz) {
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
    int k = key[i] - c.cell;                                                   \
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
          int cc = cellid(c, x, y, z), e = end[c.cell + cc];                   \
          for (int j = start[c.cell + cc]; j < e; j++) {                       \
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

__global__ void drift(int move, const int *bx, double4 *p, double4 *v, const double4 *a,
                      unsigned *key, int *idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  const Param &c = cs[bx[i]];
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
  key[i] = cellof(c, pi);
  idx[i] = i;
}

__global__ void gather(const int *idx, const double4 *p, const double4 *v, const double *S,
                       const int *id, const int *bx, double4 *p2, double4 *v2, double *S2,
                       int *id2, int *bx2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  int k = idx[i];
  p2[i] = p[k], v2[i] = v[k], S2[i] = S[k], id2[i] = id[k], bx2[i] = bx[k];
}

__global__ void ranges(const unsigned *key, int *start, int *end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  unsigned k = key[i];
  if (i == 0 || key[i - 1] != k)
    start[k] = i;
  if (i == cn - 1 || key[i + 1] != k)
    end[k] = i + 1;
}

__device__ bool owns(int i, int j) { return (i + j) & 1 ? i < j : i > j; }

__global__ void build(const int *bx, double4 *p, double4 *v, const double *S, double *d,
                      const unsigned *key, const int *start, const int *end, int cap, int *full,
                      int *nfull, int capo, int *own, int *rem, int *over) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  const Param &c = cs[bx[i]];
  double4 pi = p[i];
  double s = 0;
  int m = 0, o = 0;
  NEIGHBORS(s += W(c, sqrt(r2)); if (j != i) {
    if (m < cap)
      full[m * cn + i] = j;
    m++;
    if (cc != k && owns(i, j)) {
      if (o < capo)
        own[o * cn + i] = j;
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

__global__ void entropy(const int *bx, double *S, const double *d, const double4 *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  const Param &c = cs[bx[i]];
  S[i] = log(d[i]) + (c.mode == 2 ? c.amp * sin(c.kw * p[i].y) : 0);
}

__global__ void force(const int *bx, const Sched *sched, const double4 *p, double4 *a,
                      const int *full, const int *nfull) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn || (sched && sched[bx[i]].tau == 0))
    return;
  const Param &c = cs[bx[i]];
  double4 pi = p[i];
  double fx = 0, fy = 0, fz = 0;
  int m = nfull[i];
  for (int k = 0; k < m; k++) {
    double4 pj = p[full[k * cn + i]];
    double dx, dy, dz, r2 = dist2(c, pi, pj, &dx, &dy, &dz);
    double f = (pi.w + pj.w) * F(c, sqrt(r2));
    fx = fma(f, dx, fx), fy = fma(f, dy, fy), fz = fma(f, dz, fz);
  }
  a[i] = make_double4(fx, fy, fz, 0);
}

__global__ void kick(const int *bx, double4 *v, const double4 *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  const Param &c = cs[bx[i]];
  double4 vi = v[i], ai = a[i];
  vi.x = fma(ai.x, 0.5 * c.dt, vi.x), vi.y = fma(ai.y, 0.5 * c.dt, vi.y),
  vi.z = fma(ai.z, 0.5 * c.dt, vi.z);
  v[i] = vi;
}

__device__ double uniform(uint64_t key) { return ((mix(key) >> 11) + 0.5) * 0x1p-53; }

__device__ void ou(const Param &c, double tau, double Ti, double Tj, double a, double b,
                   double *ep, double *et, double *sp, double *st) {
  double si = 1 / (Ti + Tj), tt = Ti * Tj * si, dij = 4 * c.kB * tt * si / 3;
  double var = 4 * c.kB * tt / (1 - dij);
  *ep = exp(-2 * (1 - dij) * (4 * a / 3 + b) * tau), *et = exp(-2 * (1 - dij) * a * tau);
  *sp = sqrt(var * (1 - *ep * *ep)), *st = sqrt(var * (1 - *et * *et));
}

__device__ void pair(const Param &c, int i, int j, uint64_t seed, double tau, const double4 *p,
                     double4 *v, const double *d, const int *id) {
  double dx, dy, dz, r2 = dist2(c, p[i], p[j], &dx, &dy, &dz);
  if (r2 >= c.h2)
    return;
  double4 vi = ldcg(&v[i]), vj = ldcg(&v[j]);
  double ri = rsqrt(r2), r = r2 * ri, Ti = vi.w, Tj = vj.w;
  double f = F(c, r) * d[i] * d[j], ex = dx * ri, ey = dy * ri, ez = dz * ri;
  double a = c.ca * f, b = c.cb * f, ep, et, sp, st;
  double ux = vi.x - vj.x, uy = vi.y - vj.y, uz = vi.z - vj.z;
  double up = fma(ex, ux, fma(ey, uy, ez * uz));
  double tx = fma(-up, ex, ux), ty = fma(-up, ey, uy), tz = fma(-up, ez, uz);
  uint64_t key = mix(seed ^ c.seed ^ (uint64_t)id[i]) ^ (uint64_t)id[j] << 32;
  ou(c, tau, Ti, Tj, a, b, &ep, &et, &sp, &st);
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
    double ep1, et1, sp1, st1;
    ou(c, tau, Ti1, Tj1, a, b, &ep1, &et1, &sp1, &st1);
    double rp = fma(-ep1, up1, up) / sp1, rx = fma(-et1, tx1, tx) / st1,
           ry = fma(-et1, ty1, ty) / st1, rz = fma(-et1, tz1, tz) / st1;
    double fwd = fma(gp, gp, fma(gx, gx, fma(gy, gy, fma(gz, gz, -ge * ge))));
    double la = (1.5 / c.kB - 1) * log(Ti1 * Tj1 / (Ti * Tj)) -
                0.5 * fma(rp, rp, fma(rx, rx, fma(ry, ry, rz * rz))) + 0.5 * fwd +
                log(sp * st * st / (sp1 * st1 * st1));
    ok = log(uniform(key + 3)) < la;
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
      double s1 = sqrt(4 * c.kappa * c.kB * Ti1 * Tj1 * f * tau);
      double back = fma(2 * c.kappa * f * tau, Ti1 - Tj1, -q) / s1;
      double la = (1.5 / c.kB - 1) * log(Ti1 * Tj1 / (Ti * Tj)) - 0.5 * back * back +
                  0.5 * g2.x * g2.x + log(s / s1);
      ok = log(uniform(key + 4)) < la;
    }
#endif
    if (ok)
      Ti = Ti1, Tj = Tj1;
  }
  vi.w = Ti, vj.w = Tj;
  v[i] = vi, v[j] = vj;
}

__device__ void within(const Param &c, int s0, int m, int rot, int lane, uint64_t seed,
                       double tau, const double4 *p, double4 *v, const double *d, const int *id) {
  int M = m + (m & 1);
  for (int r0 = 0; r0 < M - 1; r0++) {
    int r = (r0 + rot) % (M - 1);
    for (int q = lane; q < M / 2; q += 32) {
      int a = q == 0 ? M - 1 : (r + q) % (M - 1), b = q == 0 ? r : (r - q + M - 1) % (M - 1);
      if (a < m && b < m)
        pair(c, s0 + a, s0 + b, seed, tau, p, v, d, id);
    }
    __syncwarp();
  }
}

__device__ void cross(const Param &c, int s0, int m, unsigned rot, int lane, uint64_t seed,
                      double tau, const double4 *p, double4 *v, const double *d, const int *id,
                      const int *own, const int *rem) {
  for (int base = 0; base < m; base += 32) {
    int l = base + lane, i = s0 + l, nj = l < m ? rem[i] : 0, k = 0, j = -1;
    int o = nj ? (nj * lane / 32 + rot) % nj : 0;
    while (__any_sync(0xffffffff, k < nj || j >= 0)) {
      if (j < 0 && k < nj) {
        int q = o + k++;
        j = own[(q < nj ? q : q - nj) * cn + i];
      }
      unsigned same = __match_any_sync(0xffffffff, j);
      if (j >= 0 && __ffs(same) - 1 == lane) {
        pair(c, i, j, seed, tau, p, v, d, id);
        j = -1;
      }
      __syncwarp();
    }
  }
}

__global__ void dissipate(int slot, const Sched *sched, const double4 *p, double4 *v,
                          const double *d, const int *id, const int *start, const int *end,
                          const int *own, const int *rem) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= wb[nbox])
    return;
  int lo = 0, hi = nbox;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (wb[mid] <= w)
      lo = mid;
    else
      hi = mid;
  }
  const Sched &sc = sched[lo];
  if (sc.tau == 0)
    return;
  const Param &c = cs[lo];
  w -= wb[lo];
  int color = sc.perm[slot], nx = c.ncx / 3, ny = c.ncy / 3;
  int cx = color % 3 + 3 * (w % nx), cy = color / 3 % 3 + 3 * (w / nx % ny),
      cz = color / 9 + 3 * (w / (nx * ny));
  int cl = cellid(c, cx, cy, cz), c0 = c.cell + cl, s0 = start[c0], m = end[c0] - s0;
  uint64_t seed = sc.seed, hsh = mix(seed ^ c.seed ^ 0x5851f42d4c957f2dULL ^ (uint64_t)cl);
  int rot = m > 2 ? hsh % (m + (m & 1) - 1) : 0;
  if (hsh >> 63) {
    within(c, s0, m, rot, lane, seed, sc.tau, p, v, d, id);
    cross(c, s0, m, (unsigned)(hsh >> 32), lane, seed, sc.tau, p, v, d, id, own, rem);
  } else {
    cross(c, s0, m, (unsigned)(hsh >> 32), lane, seed, sc.tau, p, v, d, id, own, rem);
    within(c, s0, m, rot, lane, seed, sc.tau, p, v, d, id);
  }
}

__global__ void pressure(const int *bx, const Sched *sched, double4 *p, const double4 *v,
                         const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cn && sched[bx[i]].tau != 0)
    p[i].w = v[i].w * d[i];
}

__global__ void update(const int *bx, const Sched *sched, double *S, const double4 *v,
                       const double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cn && sched[bx[i]].tau != 0)
    S[i] = 1.5 * log(v[i].w * cbrt(d[i] * d[i]));
}

#define NM 4
#define NQ (9 + 30 * NM)

__global__ void diag(const int *bx, const double4 *p, const double4 *v, const double *S,
                     double *val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cn)
    return;
  const Param &c = cs[bx[i]];
  double4 pi = p[i], vi = v[i];
  double v2 = fma(vi.x, vi.x, fma(vi.y, vi.y, vi.z * vi.z));
  double q[9] = {(c.mode == 2 ? S[i] : c.mode ? vi.y : vi.x) * sin(c.kw * pi.y),
                 fma(1.5, vi.w, 0.5 * v2), S[i], v2, vi.w, vi.w * vi.w, vi.x, vi.y, vi.z};
  for (int k = 0; k < 9; k++)
    val[k * cn + i] = q[k];
  double x[3] = {pi.x * c.Lxi, pi.y * c.Lyi, pi.z * c.Lzi}, u[3] = {vi.x, vi.y, vi.z};
  for (int a = 0; a < 3; a++)
    for (int m = 1; m <= NM; m++) {
      double sn, cs, f[5] = {u[a], u[(a + 1) % 3], u[(a + 2) % 3], 1, S[i]};
      sincospi(2 * m * x[a], &sn, &cs);
      for (int k = 0; k < 5; k++) {
        int q = 9 + 10 * (NM * a + m - 1) + 2 * k;
        val[q * cn + i] = f[k] * cs;
        val[(q + 1) * cn + i] = f[k] * sn;
      }
    }
}

static int n, nb, ncells, bits, B, nw;
static double4 *p, *v, *a, *p2, *v2;
static double *S, *S2, *d, *val, *sum;
static int *id, *id2, *bx, *bx2, *idx, *idx2, *start, *end, *full, *nfull, *own, *rem, *over,
    cap = 160, capo = 96;
static unsigned *key, *key2;
static void *tmp, *rtmp;
static size_t ntmp, nrtmp;
static int *segb, *sege;

static void sort(int move) {
  drift<<<nb, 256>>>(move, bx, p, v, a, key, idx);
  cub::DeviceRadixSort::SortPairs(tmp, ntmp, key, key2, idx, idx2, n, 0, bits);
  gather<<<nb, 256>>>(idx2, p, v, S, id, bx, p2, v2, S2, id2, bx2);
  std::swap(p, p2), std::swap(v, v2), std::swap(S, S2), std::swap(id, id2), std::swap(bx, bx2);
  cudaMemsetAsync(start, 0, ncells * sizeof *start);
  cudaMemsetAsync(end, 0, ncells * sizeof *end);
  ranges<<<nb, 256>>>(key2, start, end);
}

static void neighbors(void) {
  int o[2];
  for (;;) {
    cudaMemsetAsync(over, 0, 2 * sizeof *over);
    build<<<nb, 256>>>(bx, p, v, S, d, key2, start, end, cap, full, nfull, capo, own, rem, over);
    cudaMemcpy(o, over, sizeof o, cudaMemcpyDeviceToHost);
    if (o[0] == 0 && o[1] == 0)
      break;
    if (o[0])
      cudaFree(full), cap = o[0] + 16, cudaMalloc(&full, (size_t)cap * n * sizeof *full);
    if (o[1])
      cudaFree(own), capo = o[1] + 16, cudaMalloc(&own, (size_t)capo * n * sizeof *own);
  }
}

static Sched hsched[MAXB], *sched;
static uint64_t ticks[MAXB];

static void permute(uint64_t s, int *perm) {
  for (int k = 0; k < 27; k++)
    perm[k] = k;
  for (int k = 26; k > 0; k--) {
    int j = (s = mix(s + 0x9e3779b97f4a7c15ULL)) % (k + 1);
    std::swap(perm[k], perm[j]);
  }
}

static void relax(const double *tau, const uint64_t *cseed) {
  for (int q = 0; q < B; q++) {
    hsched[q].tau = tau[q];
    if (tau[q] > 0) {
      ticks[q]++;
      hsched[q].seed = mix(ticks[q]);
      permute(mix(ticks[q] ^ 0xd1342543de82ef95ULL ^ cseed[q]), hsched[q].perm);
    }
  }
  cudaMemcpy(sched, hsched, B * sizeof *hsched, cudaMemcpyHostToDevice);
  for (int k = 0; k < 27; k++)
    dissipate<<<(32 * nw + 127) / 128, 128>>>(k, sched, p, v, d, id, start, end, own, rem);
  update<<<nb, 256>>>(bx, sched, S, v, d);
  pressure<<<nb, 256>>>(bx, sched, p, v, d);
  force<<<nb, 256>>>(bx, sched, p, a, full, nfull);
}

int main(int argc, char **argv) {
  int i, j, k, l, b, q, step, maxsteps = 0, steps[MAXB], every[MAXB], ncell[MAXB], nyc[MAXB],
      hwb[MAXB + 1], off[MAXB], owed[MAXB] = {0};
  double tau[MAXB];
  uint64_t cseed[MAXB];
  double A = cbrt(4), lambda[MAXB];
  double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  char line[1024], name[MAXB][256];
  FILE *file[MAXB], *fk[MAXB];
  Param hc[MAXB];
  cudaEvent_t t0, t1;
  float ms;
  if (argc != 1) {
    fprintf(stderr,
            "usage: %s < list\n"
            "each line: out shear|sound|heat|rest h rows eta zeta kappa kB dt steps every "
            "[amp cube [seed]]\n",
            argv[0]);
    return 1;
  }
  n = ncells = nw = 0;
  for (B = 0; B < MAXB && fgets(line, sizeof line, stdin);) {
    char mode[64];
    int rows, cube = 0, waves;
    unsigned long long seed = 0;
    double h, amp = -1;
    Param &c = hc[B];
    if (line[0] == '#' || line[0] == '\n')
      continue;
    q = sscanf(line, "%255s %63s %lf %d %lf %lf %lf %lf %lf %d %d %lf %d %llu", name[B], mode, &h,
               &rows, &c.eta, &c.zeta, &c.kappa, &c.kB, &c.dt, &steps[B], &every[B], &amp, &cube,
               &seed);
    if (q != 11 && q != 13 && q != 14) {
      fprintf(stderr, "%s: bad line: %s", argv[0], line);
      return 1;
    }
    c.mode = mode[0] == 'h' ? 2 : mode[1] == 'o';
    c.amp = q >= 13 ? amp : mode[0] == 'r' ? 0 : 1e-3;
    c.seed = seed ? mix(seed) : 0;
    lambda[B] = rows * A / 2;
    waves = (int)ceil(3 * h / lambda[B]);
    if (rows * waves % 2)
      waves++;
    nyc[B] = rows * waves / 2;
    ncell[B] = (int)ceil((cube != 0 && lambda[B] > 3 * h ? lambda[B] : 3 * h) / A);
    c.h = h, c.h2 = h * h, c.hinv = 1 / h;
    c.ca = 5 * c.eta / 3 - c.zeta, c.cb = 16 * c.zeta / 3 + 10 * c.eta / 9;
    c.Lx = c.Lz = ncell[B] * A;
    c.Ly = nyc[B] * A;
    c.Lxi = 1 / c.Lx, c.Lyi = 1 / c.Ly, c.Lzi = 1 / c.Lz;
    c.n = 4 * ncell[B] * ncell[B] * nyc[B];
    c.ncx = c.ncz = 3 * (int)(c.Lx / h / 3);
    c.ncy = 3 * (int)(c.Ly / h / 3);
#ifndef LUCY
    c.cw = 21 / (2 * M_PI * h * h * h);
    c.cf = 210 / (M_PI * h * h * h * h * h);
#else
    c.cw = 105 / (16 * M_PI * h * h * h);
    c.cf = 315 / (4 * M_PI * h * h * h * h * h);
#endif
    c.kw = 2 * M_PI / lambda[B];
    c.cell = ncells;
    ncells += c.ncx * c.ncy * c.ncz;
    hwb[B] = nw;
    nw += c.ncx / 3 * (c.ncy / 3) * (c.ncz / 3);
    off[B] = n;
    n += c.n;
    if (steps[B] > maxsteps)
      maxsteps = steps[B];
    if ((file[B] = fopen(name[B], "w")) == NULL) {
      fprintf(stderr, "%s: cannot write %s\n", argv[0], name[B]);
      return 1;
    }
    fprintf(file[B], "# lambda/h %.6g k %.10g n %d\n", lambda[B] / h, c.kw, c.n);
    snprintf(line, sizeof line, "%s.k", name[B]);
    if ((fk[B] = fopen(line, "w")) == NULL) {
      fprintf(stderr, "%s: cannot write %s\n", argv[0], line);
      return 1;
    }
    fprintf(fk[B], "# k1 %.10g %.10g %.10g n %d; t, then for directions x, y, z and m = 1..%d: "
                   "v along, v across (two, cyclic), 1, S times cos and sin of m k1 r, summed and "
                   "divided by n\n", 2 * M_PI / c.Lx, 2 * M_PI / c.Ly, 2 * M_PI / c.Lz, c.n, NM);
    B++;
  }
  if (fgets(line, sizeof line, stdin)) {
    fprintf(stderr, "%s: more than %d boxes\n", argv[0], MAXB);
    return 1;
  }
  hwb[B] = nw;
  for (q = 0; q < B; q++)
    cseed[q] = hc[q].seed;
  cudaMalloc(&sched, B * sizeof *sched);
  cudaMemcpyToSymbol(cs, hc, B * sizeof *hc);
  cudaMemcpyToSymbol(cn, &n, sizeof n);
  cudaMemcpyToSymbol(nbox, &B, sizeof B);
  cudaMemcpyToSymbol(wb, hwb, (B + 1) * sizeof *hwb);
  nb = (n + 255) / 256;
  for (bits = 1; (1 << bits) < ncells; bits++)
    ;

  double4 *hp = (double4 *)malloc(n * sizeof *hp), *hv = (double4 *)malloc(n * sizeof *hv);
  double *hS = (double *)malloc(n * sizeof *hS), *hout = (double *)malloc(NQ * B * sizeof *hout);
  int *hid = (int *)malloc(n * sizeof *hid), *hbx = (int *)malloc(n * sizeof *hbx);
  for (l = 0, q = 0; q < B; q++)
    for (k = 0; k < ncell[q]; k++)
      for (j = 0; j < nyc[q]; j++)
        for (i = 0; i < ncell[q]; i++)
          for (b = 0; b < 4; b++, l++) {
            const Param &c = hc[q];
            double y = (j + basis[b][1]) * A, u = c.amp * sin(c.kw * y);
            hp[l] = make_double4((i + basis[b][0]) * A, y, (k + basis[b][2]) * A, 0);
            hv[l] = make_double4(c.mode == 0 ? u : 0, c.mode == 1 ? u : 0, 0, 0);
            hS[l] = 0;
            hid[l] = l - off[q];
            hbx[l] = q;
          }
  cudaMalloc(&p, n * sizeof *p), cudaMalloc(&p2, n * sizeof *p2);
  cudaMalloc(&v, n * sizeof *v), cudaMalloc(&v2, n * sizeof *v2);
  cudaMalloc(&a, n * sizeof *a);
  cudaMalloc(&S, n * sizeof *S), cudaMalloc(&S2, n * sizeof *S2), cudaMalloc(&d, n * sizeof *d);
  cudaMalloc(&id, n * sizeof *id), cudaMalloc(&id2, n * sizeof *id2);
  cudaMalloc(&bx, n * sizeof *bx), cudaMalloc(&bx2, n * sizeof *bx2);
  cudaMalloc(&idx, n * sizeof *idx), cudaMalloc(&idx2, n * sizeof *idx2);
  cudaMalloc(&key, n * sizeof *key), cudaMalloc(&key2, n * sizeof *key2);
  cudaMalloc(&start, ncells * sizeof *start), cudaMalloc(&end, ncells * sizeof *end);
  cudaMalloc(&val, (size_t)NQ * n * sizeof *val), cudaMalloc(&sum, NQ * B * sizeof *sum);
  {
    int *hb = (int *)malloc(NQ * B * sizeof *hb), *he = (int *)malloc(NQ * B * sizeof *he);
    for (k = 0; k < NQ; k++)
      for (q = 0; q < B; q++)
        hb[k * B + q] = k * n + off[q], he[k * B + q] = k * n + off[q] + hc[q].n;
    cudaMalloc(&segb, NQ * B * sizeof *segb), cudaMalloc(&sege, NQ * B * sizeof *sege);
    cudaMemcpy(segb, hb, NQ * B * sizeof *hb, cudaMemcpyHostToDevice);
    cudaMemcpy(sege, he, NQ * B * sizeof *he, cudaMemcpyHostToDevice);
    cub::DeviceSegmentedReduce::Sum(rtmp, nrtmp, val, sum, NQ * B, segb, sege);
    cudaMalloc(&rtmp, nrtmp);
  }
  cudaMalloc(&full, (size_t)cap * n * sizeof *full), cudaMalloc(&nfull, n * sizeof *nfull);
  cudaMalloc(&own, (size_t)capo * n * sizeof *own), cudaMalloc(&rem, n * sizeof *rem);
  cudaMalloc(&over, 2 * sizeof *over);
  cub::DeviceRadixSort::SortPairs(tmp, ntmp, key, key2, idx, idx2, n, 0, bits);
  cudaMalloc(&tmp, ntmp);
  cudaMemcpy(p, hp, n * sizeof *p, cudaMemcpyHostToDevice);
  cudaMemcpy(v, hv, n * sizeof *v, cudaMemcpyHostToDevice);
  cudaMemcpy(S, hS, n * sizeof *S, cudaMemcpyHostToDevice);
  cudaMemcpy(id, hid, n * sizeof *id, cudaMemcpyHostToDevice);
  cudaMemcpy(bx, hbx, n * sizeof *bx, cudaMemcpyHostToDevice);

  sort(0);
  neighbors();
  entropy<<<nb, 256>>>(bx, S, d, p);
  neighbors();
  force<<<nb, 256>>>(bx, NULL, p, a, full, nfull);

  cudaEventCreate(&t0), cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (step = 0; step <= maxsteps; step++) {
    int due = 0, sync = 0;
    for (q = 0; q < B; q++) {
      int out = step <= steps[q] && step % every[q] == 0;
      due |= out;
      tau[q] = out && owed[q] ? hc[q].dt / 2 : 0;
      sync |= tau[q] > 0;
      if (out)
        owed[q] = 0;
    }
    if (sync)
      relax(tau, cseed);
    if (due) {
      diag<<<nb, 256>>>(bx, p, v, S, val);
      cub::DeviceSegmentedReduce::Sum(rtmp, nrtmp, val, sum, NQ * B, segb, sege);
      cudaMemcpy(hout, sum, NQ * B * sizeof *sum, cudaMemcpyDeviceToHost);
      for (q = 0; q < B; q++)
        if (step <= steps[q] && step % every[q] == 0) {
          double s[NQ];
          int m = hc[q].n;
          for (k = 0; k < NQ; k++)
            s[k] = hout[k * B + q];
          fprintf(file[q], "%.10g %.10g %.15g %.15g %.10g %.10g %.10g %.3g %.3g %.3g\n",
                  step * hc[q].dt, 2 * s[0] / m, s[1], s[2], s[3] / (3 * m), hc[q].kB * s[4] / m,
                  s[5] * m / (s[4] * s[4]) - 1, s[6], s[7], s[8]);
          fprintf(fk[q], "%.10g", step * hc[q].dt);
          for (k = 9; k < NQ; k++)
            fprintf(fk[q], " %.10g", s[k] / m);
          fputc('\n', fk[q]);
        }
    }
    if (step == maxsteps)
      break;
    for (q = 0; q < B; q++)
      tau[q] = owed[q] ? hc[q].dt : hc[q].dt / 2, owed[q] = 1;
    relax(tau, cseed);
    sort(1);
    neighbors();
    force<<<nb, 256>>>(bx, NULL, p, a, full, nfull);
    kick<<<nb, 256>>>(bx, v, a);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  cudaEventElapsedTime(&ms, t0, t1);
  for (q = 0; q < B; q++)
    fclose(file[q]), fclose(fk[q]);
  fprintf(stderr, "%s: boxes %d n %d steps %d time %.3f s %.3g particle-steps/s\n", argv[0], B, n,
          maxsteps, ms / 1e3, (double)n * maxsteps / (ms / 1e3));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", argv[0], cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
