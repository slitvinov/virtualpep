#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int n, ncx, ncy, *head, *next;
static double h, Lx, Ly, eta, zeta, kappa, kB, dt;
static uint64_t tick;
static double *x, *y, *vx, *vy, *S, *d, *T, *P, *ax, *ay;

#ifndef LUCY
static double W(double r) {
  double q = r / h, s = 1 - q;
  return 7 / (M_PI * h * h) * (1 + 4 * q) * s * s * s * s;
}

static double F(double r) {
  double s = 1 - r / h;
  return 140 / (M_PI * h * h * h * h) * s * s * s;
}
#else
static double W(double r) {
  double q = r / h;
  return 5 / (M_PI * h * h) * (1 + 3 * q) * (1 - q) * (1 - q) * (1 - q);
}

static double F(double r) {
  double q = r / h;
  return 60 / (M_PI * h * h * h * h) * (1 - q) * (1 - q);
}
#endif

static uint64_t mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
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

static void cells(void) {
  int i, c;
  for (c = 0; c < ncx * ncy; c++)
    head[c] = -1;
  for (i = 0; i < n; i++) {
    c = cellx(i) + ncx * celly(i);
    next[i] = head[c];
    head[c] = i;
  }
}

#define NEIGHBORS(...)                                                         \
  {                                                                            \
    int cx = cellx(i), cy = celly(i);                                          \
    for (int oy = -1; oy <= 1; oy++)                                           \
      for (int ox = -1; ox <= 1; ox++) {                                       \
        int c = (cx + ox + ncx) % ncx + ncx * ((cy + oy + ncy) % ncy);         \
        for (int j = head[c]; j != -1; j = next[j]) {                          \
          double dx = image(x[i] - x[j], Lx), dy = image(y[i] - y[j], Ly);     \
          double r = sqrt(dx * dx + dy * dy);                                  \
          if (r < h) {                                                         \
            __VA_ARGS__                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
  }

static void forces(void) {
  int i;
  cells();
#pragma omp parallel for
  for (i = 0; i < n; i++) {
    double s = 0;
    NEIGHBORS(s += W(r);)
    d[i] = s;
    T[i] = d[i] * exp(S[i]);
    P[i] = d[i] * T[i];
  }
#pragma omp parallel for
  for (i = 0; i < n; i++) {
    double fx = 0, fy = 0;
    NEIGHBORS(if (j != i) {
      double p = (P[i] / (d[i] * d[i]) + P[j] / (d[j] * d[j])) * F(r);
      fx += p * dx;
      fy += p * dy;
    })
    ax[i] = fx;
    ay[i] = fy;
  }
}

static double relax(double u, double rate, double var, double z) {
  double e = exp(-rate * dt);
  return u * e + sqrt(rate > 0 ? var / (2 * rate) * (1 - e * e) : var * dt) * z;
}

static void pair(int i, int j) {
  double dx = image(x[i] - x[j], Lx), dy = image(y[i] - y[j], Ly);
  double r = sqrt(dx * dx + dy * dy);
  if (r >= h)
    return;
  double f = F(r) / (d[i] * d[j]), ex = dx / r, ey = dy / r;
  double a = (2 * eta - zeta) * f, b = 4 * zeta * f;
  double tt = T[i] * T[j] / (T[i] + T[j]), dij = 2 * kB * tt / (T[i] + T[j]);
  double A2 = 8 * kB * a * tt, B2 = 8 * kB * b * tt;
  double ux = vx[i] - vx[j], uy = vy[i] - vy[j];
  double up = ex * ux + ey * uy, ut = ey * ux - ex * uy;
  uint64_t key = mix(mix(tick) ^ (uint64_t)i) ^ (uint64_t)j << 32;
  double up1 = relax(up, 2 * (1 - dij) * (a + b), 4 * (A2 + (B2 - A2) / 2),
                     gauss(key));
  double ut1 = relax(ut, 2 * (1 - dij) * a, 2 * A2, gauss(key + 1));
  double wx = (up1 * ex + ut1 * ey - ux) / 2, wy = (up1 * ey - ut1 * ex - uy) / 2;
  double dK = (up1 * up1 + ut1 * ut1 - up * up - ut * ut) / 4;
  double q = -2 * kappa * f * (T[i] - T[j]) * dt +
             sqrt(4 * kappa * kB * T[i] * T[j] * f * dt) * gauss(key + 2);
  vx[i] += wx, vy[i] += wy, vx[j] -= wx, vy[j] -= wy;
  T[i] += q - dK / 2;
  T[j] += -q - dK / 2;
}

static void dissipate(void) {
  int ox[] = {1, -1, 0, 1}, oy[] = {0, 1, 1, 1};
  tick++;
  for (int color = 0; color < 9; color++) {
#pragma omp parallel for collapse(2)
    for (int cy = color / 3; cy < ncy; cy += 3)
      for (int cx = color % 3; cx < ncx; cx += 3) {
        int c = cx + ncx * cy;
        for (int i = head[c]; i != -1; i = next[i]) {
          for (int j = next[i]; j != -1; j = next[j])
            pair(i, j);
          for (int k = 0; k < 4; k++) {
            int c2 = (cx + ox[k] + ncx) % ncx + ncx * ((cy + oy[k] + ncy) % ncy);
            for (int j = head[c2]; j != -1; j = next[j])
              pair(i, j);
          }
        }
      }
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    S[i] = log(T[i] / d[i]);
}

int main(int argc, char **argv) {
  int mode, rows, waves, nrow, ncol, i, j, k, step, steps, every, dump;
  double s, hy, lambda, kw, amp, u, E, Stot, v2, Tm;
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
  eta = atof(argv[4]);
  zeta = atof(argv[5]);
  kappa = atof(argv[6]);
  kB = atof(argv[7]);
  dt = atof(argv[8]);
  steps = atoi(argv[9]);
  every = atoi(argv[10]);
  dump = argc == 13 ? atoi(argv[12]) : 0;

  s = sqrt(2 / sqrt(3));
  hy = s * sqrt(3) / 2;
  lambda = rows * hy;
  waves = (int)ceil(3 * h / lambda);
  if (rows * waves % 2)
    waves++;
  nrow = rows * waves;
  ncol = (int)ceil((dump > 0 && lambda > 3 * h ? lambda : 3 * h) / s);
  Lx = ncol * s;
  Ly = nrow * hy;
  n = ncol * nrow;
  ncx = 3 * (int)(Lx / h / 3);
  ncy = 3 * (int)(Ly / h / 3);
  kw = 2 * M_PI / lambda;
  amp = argc == 13 ? atof(argv[11]) : argv[1][0] == 'r' ? 0 : 1e-3;

  x = malloc(n * sizeof *x), y = malloc(n * sizeof *y);
  vx = malloc(n * sizeof *vx), vy = malloc(n * sizeof *vy);
  S = malloc(n * sizeof *S), d = malloc(n * sizeof *d);
  T = malloc(n * sizeof *T), P = malloc(n * sizeof *P);
  ax = malloc(n * sizeof *ax), ay = malloc(n * sizeof *ay);
  next = malloc(n * sizeof *next), head = malloc(ncx * ncy * sizeof *head);

  for (k = 0, j = 0; j < nrow; j++)
    for (i = 0; i < ncol; i++, k++) {
      x[k] = wrap((i + 0.5 * (j % 2)) * s, Lx);
      y[k] = j * hy;
      vx[k] = mode ? 0 : amp * sin(kw * y[k]);
      vy[k] = mode ? amp * sin(kw * y[k]) : 0;
      S[k] = 0;
    }
  forces();
  for (i = 0; i < n; i++)
    S[i] = -log(d[i]);
  forces();

  printf("# lambda/h %.6g k %.10g n %d\n", lambda / h, kw, n);
  for (step = 0; step <= steps; step++) {
    if (dump > 0 && step % dump == 0) {
      char name[64];
      FILE *file;
      snprintf(name, sizeof name, "dump.%06d", step / dump);
      file = fopen(name, "w");
      fprintf(file, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n0 %.10g\n0 %.10g\n-0.5 0.5\nITEM: ATOMS x y z vx vy T\n", step, n, Lx, Ly);
      for (i = 0; i < n; i++)
        fprintf(file, "%.6g %.6g 0 %.6g %.6g %.6g\n", x[i], y[i], vx[i], vy[i], T[i]);
      fclose(file);
    }
    if (step % every == 0) {
      for (u = 0, E = 0, Stot = 0, v2 = 0, Tm = 0, i = 0; i < n; i++) {
        u += (mode ? vy[i] : vx[i]) * sin(kw * y[i]);
        E += (vx[i] * vx[i] + vy[i] * vy[i]) / 2 + T[i];
        Stot += S[i];
        v2 += vx[i] * vx[i] + vy[i] * vy[i];
        Tm += T[i];
      }
      printf("%.10g %.10g %.15g %.15g %.10g %.10g\n", step * dt, 2 * u / n, E,
             Stot, v2 / (2 * n), kB * Tm / n);
    }
#pragma omp parallel for
    for (i = 0; i < n; i++) {
      vx[i] += ax[i] * dt / 2;
      vy[i] += ay[i] * dt / 2;
      x[i] = wrap(x[i] + vx[i] * dt, Lx);
      y[i] = wrap(y[i] + vy[i] * dt, Ly);
    }
    forces();
#pragma omp parallel for
    for (i = 0; i < n; i++) {
      vx[i] += ax[i] * dt / 2;
      vy[i] += ay[i] * dt / 2;
    }
    dissipate();
  }
  return 0;
}
