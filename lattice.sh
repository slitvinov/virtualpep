# deterministic linear modes against lattice3; batch list on stdout
dt=$1
awk -v dt=$dt '{
  n = int(40 / dt + 0.5)
  for (m = 0; m < 3; m++) {
    mode = m == 0 ? "shear" : m == 1 ? "sound" : "heat"
    printf "%s.%d.%d.%s.%s %s %d %d %s %s %s 0 %s %d %d 1e-6 0\n", mode, $1, $2, $4, dt, mode, $1, $2, $4, $5, $6, dt, n, int(0.05 / dt + 0.5)
  }
}' ${2:-lattice3}
