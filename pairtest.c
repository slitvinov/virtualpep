#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static double C = 1.5, kB, ca, cb, kappa, f, dt;
static uint64_t state = 1;

static uint64_t next(void) {
  uint64_t z = state += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static double uniform(void) { return ((next() >> 11) + 0.5) * 0x1p-53; }

static double gauss(void) {
  return sqrt(-2 * log(uniform())) * cos(2 * M_PI * uniform());
}

static double lognu(double ei, double ej) {
  return (C / kB - 1) * (log(ei) + log(ej));
}

static void coef(double ei, double ej, double *ap, double *at, double *sp, double *st) {
  double Ti = ei / C, Tj = ej / C, tt = Ti * Tj / (Ti + Tj);
  double d = 2 * kB * Ti * Tj / ((Ti + Tj) * (Ti + Tj) * C), V = 4 * kB * tt / (1 - d);
  *ap = exp(-2 * (1 - d) * (4 * ca / 3 + cb) * f * dt);
  *at = exp(-2 * (1 - d) * ca * f * dt);
  *sp = sqrt(V * (1 - *ap * *ap));
  *st = sqrt(V * (1 - *at * *at));
}

static int friction(int metro, double *u, double *ei, double *ej) {
  double ap, at, sp, st, ap1, at1, sp1, st1, G[3], u1[3], w[3], du2 = 0, r = 0, g2 = 0;
  coef(*ei, *ej, &ap, &at, &sp, &st);
  for (int c = 0; c < 3; c++)
    G[c] = gauss(), g2 += G[c] * G[c];
  u1[0] = ap * u[0] + sp * G[0];
  u1[1] = at * u[1] + st * G[1];
  u1[2] = at * u[2] + st * G[2];
  for (int c = 0; c < 3; c++)
    du2 += u1[c] * u1[c] - u[c] * u[c];
  double e1 = *ei - du2 / 8, e2 = *ej - du2 / 8;
  if (metro) {
    if (e1 <= 0 || e2 <= 0)
      return 0;
    coef(e1, e2, &ap1, &at1, &sp1, &st1);
    w[0] = (u[0] - ap1 * u1[0]) / sp1;
    w[1] = (u[1] - at1 * u1[1]) / st1;
    w[2] = (u[2] - at1 * u1[2]) / st1;
    for (int c = 0; c < 3; c++)
      r += w[c] * w[c];
    double la = lognu(e1, e2) - lognu(*ei, *ej) - r / 2 - log(sp1 * st1 * st1) + g2 / 2 +
                log(sp * st * st);
    if (log(uniform()) > la)
      return 0;
  }
  for (int c = 0; c < 3; c++)
    u[c] = u1[c];
  *ei = e1, *ej = e2;
  return 1;
}

static double drift(double ei, double ej) { return -2 * kappa * f * (ei - ej) / C * dt; }
static double spread(double ei, double ej) { return sqrt(4 * kappa * kB * f * ei * ej / (C * C) * dt); }

static int heat(int metro, double *ei, double *ej) {
  double G = gauss(), s = spread(*ei, *ej), q = drift(*ei, *ej) + s * G;
  double e1 = *ei + q, e2 = *ej - q;
  if (metro) {
    if (e1 <= 0 || e2 <= 0)
      return 0;
    double s1 = spread(e1, e2), back = *ei - e1 - drift(e1, e2);
    double la = lognu(e1, e2) - lognu(*ei, *ej) - back * back / (2 * s1 * s1) - log(s1) +
                G * G / 2 + log(s);
    if (log(uniform()) > la)
      return 0;
  }
  *ei = e1, *ej = e2;
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s metro kB dt kappa steps\n", argv[0]);
    return 1;
  }
  int metro = atoi(argv[1]);
  kB = atof(argv[2]), dt = atof(argv[3]), kappa = atof(argv[4]);
  long steps = atol(argv[5]), nb = 100, k;
  ca = 5.0 / 3, cb = 10.0 / 9, f = 1;
  double u[3] = {0, 0, 0}, ei = 1.5, ej = 1.5, E = ei + ej, acc = 0;
  double mK[100] = {0}, mE[100] = {0}, mE2[100] = {0};
  for (k = 0; k < steps / 10; k++)
    friction(metro, u, &ei, &ej), heat(metro, &ei, &ej);
  for (k = 0; k < steps; k++) {
    acc += friction(metro, u, &ei, &ej);
    acc += heat(metro, &ei, &ej);
    double K = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / 4;
    int b = k * nb / steps;
    mK[b] += K / E, mE[b] += ei / E, mE2[b] += ei * ei / (E * E);
  }
  double a = C / kB, A = 1.5 + 2 * a, K0 = 1.5 / A, e0 = a / A, v0 = a * (A - a) / (A * A * (A + 1));
  double sK = 0, sK2 = 0, sV = 0, sV2 = 0;
  for (int b = 0; b < nb; b++) {
    double K = mK[b] * nb / steps, e = mE[b] * nb / steps, var = mE2[b] * nb / steps - e * e;
    sK += K, sK2 += K * K, sV += var, sV2 += var * var;
  }
  sK /= nb, sV /= nb;
  printf("%s kB %g dt %g kappa %g: <K>/E %.5f +- %.5f exact %.5f | var(ei)/E^2 %.6f +- %.6f exact %.6f | acceptance %.4f\n",
         metro ? "metropolis" : "ssa       ", kB, dt, kappa, sK, sqrt((sK2 / nb - sK * sK) / (nb - 1)), K0,
         sV, sqrt((sV2 / nb - sV * sV) / (nb - 1)), v0, acc / (2 * steps));
  (void)e0;
  return 0;
}
