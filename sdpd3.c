#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int n, ncx, ncy, ncz, *head, *next;
static double h, Lx, Ly, Lz, eta, zeta, kappa, kB, dt;
static uint64_t tick;
static double *x, *y, *z, *vx, *vy, *vz, *S, *d, *T, *P, *ax, *ay, *az;

#ifdef WENDLAND
static double W(double r) {
  double q = r / h, s = 1 - q;
  return 21 / (2 * M_PI * h * h * h) * (1 + 4 * q) * s * s * s * s;
}

static double F(double r) {
  double s = 1 - r / h;
  return 210 / (M_PI * h * h * h * h * h) * s * s * s;
}
#else
static double W(double r) {
  double q = r / h;
  return 105 / (16 * M_PI * h * h * h) * (1 + 3 * q) * (1 - q) * (1 - q) *
         (1 - q);
}

static double F(double r) {
  double q = r / h;
  return 315 / (4 * M_PI * h * h * h * h * h) * (1 - q) * (1 - q);
}
#endif

static uint64_t mix(uint64_t w) {
  w = (w ^ (w >> 30)) * 0xbf58476d1ce4e5b9ULL;
  w = (w ^ (w >> 27)) * 0x94d049bb133111ebULL;
  return w ^ (w >> 31);
}

static double gauss(uint64_t key) {
  uint64_t a = mix(key), b = mix(a ^ 0x9e3779b97f4a7c15ULL);
  double u = ((a >> 11) + 0.5) * 0x1p-53, w = (b >> 11) * 0x1p-53;
  return sqrt(-2 * log(u)) * cos(2 * M_PI * w);
}

static double wrap(double a, double L) { return a - L * floor(a / L); }
static double image(double a, double L) { return a - L * round(a / L); }
static int cellx(int i) { return (int)(x[i] / Lx * ncx) % ncx; }
static int celly(int i) { return (int)(y[i] / Ly * ncy) % ncy; }
static int cellz(int i) { return (int)(z[i] / Lz * ncz) % ncz; }
static int cellid(int cx, int cy, int cz) {
  return (cx + ncx) % ncx + ncx * ((cy + ncy) % ncy + ncy * ((cz + ncz) % ncz));
}

static void cells(void) {
  int i, c;
  for (c = 0; c < ncx * ncy * ncz; c++)
    head[c] = -1;
  for (i = 0; i < n; i++) {
    c = cellid(cellx(i), celly(i), cellz(i));
    next[i] = head[c];
    head[c] = i;
  }
}

#define NEIGHBORS(...)                                                         \
  {                                                                            \
    int cx = cellx(i), cy = celly(i), cz = cellz(i);                           \
    for (int oz = -1; oz <= 1; oz++)                                           \
      for (int oy = -1; oy <= 1; oy++)                                         \
        for (int ox = -1; ox <= 1; ox++) {                                     \
          int c = cellid(cx + ox, cy + oy, cz + oz);                           \
          for (int j = head[c]; j != -1; j = next[j]) {                        \
            double dx = image(x[i] - x[j], Lx), dy = image(y[i] - y[j], Ly),   \
                   dz = image(z[i] - z[j], Lz);                                \
            double r = sqrt(dx * dx + dy * dy + dz * dz);                      \
            if (r < h) {                                                       \
              __VA_ARGS__                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
  }

static void forces(void) {
  int i;
  cells();
#pragma omp parallel for
  for (i = 0; i < n; i++) {
    double s = 0;
    NEIGHBORS(s += W(r);)
    d[i] = s;
    T[i] = cbrt(d[i] * d[i]) * exp(2 * S[i] / 3);
    P[i] = d[i] * T[i];
  }
#pragma omp parallel for
  for (i = 0; i < n; i++) {
    double fx = 0, fy = 0, fz = 0;
    NEIGHBORS(if (j != i) {
      double p = (P[i] / (d[i] * d[i]) + P[j] / (d[j] * d[j])) * F(r);
      fx += p * dx;
      fy += p * dy;
      fz += p * dz;
    })
    ax[i] = fx;
    ay[i] = fy;
    az[i] = fz;
  }
}

static void pair(int i, int j) {
  double dx = image(x[i] - x[j], Lx), dy = image(y[i] - y[j], Ly),
         dz = image(z[i] - z[j], Lz);
  double r = sqrt(dx * dx + dy * dy + dz * dz);
  if (r >= h)
    return;
  double f = F(r) / (d[i] * d[j]), ex = dx / r, ey = dy / r, ez = dz / r;
  double a = (5 * eta / 3 - zeta) * f, b = (16 * zeta / 3 + 10 * eta / 9) * f;
  double tt = T[i] * T[j] / (T[i] + T[j]), dij = 4 * kB * tt / (3 * (T[i] + T[j]));
  double A2 = 8 * kB * a * tt, B2 = 12 * kB * b * tt;
  double ux = vx[i] - vx[j], uy = vy[i] - vy[j], uz = vz[i] - vz[j];
  double up = ex * ux + ey * uy + ez * uz;
  double tx = ux - up * ex, ty = uy - up * ey, tz = uz - up * ez;
  uint64_t key = mix(mix(tick) ^ (uint64_t)i) ^ (uint64_t)j << 32;
  double lp = 2 * (1 - dij) * (a / 3 + b) + 2 * (1 - dij) * a, lt = 2 * (1 - dij) * a;
  double vp = 4 * (A2 + (B2 - A2) / 3), vt = 2 * A2;
  double ep = exp(-lp * dt), et = exp(-lt * dt);
  double sp = sqrt(lp > 0 ? vp / (2 * lp) * (1 - ep * ep) : vp * dt);
  double st = sqrt(lt > 0 ? vt / (2 * lt) * (1 - et * et) : vt * dt);
  double gx = gauss(key), gy = gauss(key + 1), gz = gauss(key + 2);
  double ge = gx * ex + gy * ey + gz * ez;
  double up1 = up * ep + sp * gauss(key + 3);
  double tx1 = tx * et + st * (gx - ge * ex), ty1 = ty * et + st * (gy - ge * ey),
         tz1 = tz * et + st * (gz - ge * ez);
  double wx = (up1 * ex + tx1 - ux) / 2, wy = (up1 * ey + ty1 - uy) / 2,
         wz = (up1 * ez + tz1 - uz) / 2;
  double dK = (up1 * up1 + tx1 * tx1 + ty1 * ty1 + tz1 * tz1 - ux * ux - uy * uy -
               uz * uz) / 4;
  double q = -2 * kappa * f * (T[i] - T[j]) * dt +
             sqrt(4 * kappa * kB * T[i] * T[j] * f * dt) * gauss(key + 4);
  vx[i] += wx, vy[i] += wy, vz[i] += wz;
  vx[j] -= wx, vy[j] -= wy, vz[j] -= wz;
  T[i] += (q - dK / 2) / 1.5;
  T[j] += (-q - dK / 2) / 1.5;
}

static void dissipate(void) {
  int off[13][3], m = 0;
  for (int oz = -1; oz <= 1; oz++)
    for (int oy = -1; oy <= 1; oy++)
      for (int ox = -1; ox <= 1; ox++)
        if (oz > 0 || (oz == 0 && (oy > 0 || (oy == 0 && ox > 0))))
          off[m][0] = ox, off[m][1] = oy, off[m][2] = oz, m++;
  tick++;
  for (int color = 0; color < 27; color++) {
#pragma omp parallel for collapse(3)
    for (int cz = color / 9; cz < ncz; cz += 3)
      for (int cy = color / 3 % 3; cy < ncy; cy += 3)
        for (int cx = color % 3; cx < ncx; cx += 3) {
          int c = cellid(cx, cy, cz);
          for (int i = head[c]; i != -1; i = next[i]) {
            for (int j = next[i]; j != -1; j = next[j])
              pair(i, j);
            for (int k = 0; k < 13; k++) {
              int c2 = cellid(cx + off[k][0], cy + off[k][1], cz + off[k][2]);
              for (int j = head[c2]; j != -1; j = next[j])
                pair(i, j);
            }
          }
        }
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    S[i] = 1.5 * log(T[i] / cbrt(d[i] * d[i]));
}

int main(int argc, char **argv) {
  int mode, rows, waves, ncell, nyc, i, j, k, l, b, step, steps, every;
  double a, lambda, kw, amp, u, E, Stot, v2, Tm;
  double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  if (argc != 11) {
    fprintf(stderr,
            "usage: %s shear|sound|rest h rows eta zeta kappa kB dt steps every\n",
            argv[0]);
    return 1;
  }
  mode = argv[1][1] == 'o';
  h = atof(argv[2]);
  rows = atoi(argv[3]);
  eta = atof(argv[4]);
  zeta = atof(argv[5]);
  kappa = atof(argv[6]);
  kB = atof(argv[7]);
  dt = atof(argv[8]);
  steps = atoi(argv[9]);
  every = atoi(argv[10]);

  a = cbrt(4);
  lambda = rows * a / 2;
  waves = (int)ceil(3 * h / lambda);
  if (rows * waves % 2)
    waves++;
  nyc = rows * waves / 2;
  ncell = (int)ceil(3 * h / a);
  Lx = Lz = ncell * a;
  Ly = nyc * a;
  n = 4 * ncell * ncell * nyc;
  ncx = ncz = 3 * (int)(Lx / h / 3);
  ncy = 3 * (int)(Ly / h / 3);
  kw = 2 * M_PI / lambda;
  amp = argv[1][0] == 'r' ? 0 : 1e-3;

  x = malloc(n * sizeof *x), y = malloc(n * sizeof *y), z = malloc(n * sizeof *z);
  vx = malloc(n * sizeof *vx), vy = malloc(n * sizeof *vy);
  vz = malloc(n * sizeof *vz);
  S = malloc(n * sizeof *S), d = malloc(n * sizeof *d);
  T = malloc(n * sizeof *T), P = malloc(n * sizeof *P);
  ax = malloc(n * sizeof *ax), ay = malloc(n * sizeof *ay);
  az = malloc(n * sizeof *az);
  next = malloc(n * sizeof *next);
  head = malloc(ncx * ncy * ncz * sizeof *head);

  for (l = 0, k = 0; k < ncell; k++)
    for (j = 0; j < nyc; j++)
      for (i = 0; i < ncell; i++)
        for (b = 0; b < 4; b++, l++) {
          x[l] = (i + basis[b][0]) * a;
          y[l] = (j + basis[b][1]) * a;
          z[l] = (k + basis[b][2]) * a;
          vx[l] = mode ? 0 : amp * sin(kw * y[l]);
          vy[l] = mode ? amp * sin(kw * y[l]) : 0;
          vz[l] = 0;
          S[l] = 0;
        }
  forces();
  for (i = 0; i < n; i++)
    S[i] = -log(d[i]);
  forces();

  printf("# lambda/h %.6g k %.10g n %d\n", lambda / h, kw, n);
  for (step = 0; step <= steps; step++) {
    if (step % every == 0) {
      for (u = 0, E = 0, Stot = 0, v2 = 0, Tm = 0, i = 0; i < n; i++) {
        u += (mode ? vy[i] : vx[i]) * sin(kw * y[i]);
        E += (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) / 2 + 1.5 * T[i];
        Stot += S[i];
        v2 += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        Tm += T[i];
      }
      printf("%.10g %.10g %.15g %.15g %.10g %.10g\n", step * dt, 2 * u / n, E,
             Stot, v2 / (3 * n), kB * Tm / n);
    }
#pragma omp parallel for
    for (i = 0; i < n; i++) {
      vx[i] += ax[i] * dt / 2;
      vy[i] += ay[i] * dt / 2;
      vz[i] += az[i] * dt / 2;
      x[i] = wrap(x[i] + vx[i] * dt, Lx);
      y[i] = wrap(y[i] + vy[i] * dt, Ly);
      z[i] = wrap(z[i] + vz[i] * dt, Lz);
    }
    forces();
#pragma omp parallel for
    for (i = 0; i < n; i++) {
      vx[i] += ax[i] * dt / 2;
      vy[i] += ay[i] * dt / 2;
      vz[i] += az[i] * dt / 2;
    }
    dissipate();
  }
  return 0;
}
