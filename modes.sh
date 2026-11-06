# decay of the heat and sound modes and temperature fluctuations, sdpd3cu
for l in 4 6 8 12 16; do
  rows=$(awk -v l=$l 'BEGIN {printf "%d", l * 3 * 2 / exp(log(4) / 3) + 0.5}')
  tmax=$(awk -v l=$l 'BEGIN {k = 2 * 3.14159265 / (3 * l); print int(3 / (k * k / 2.5) / 0.01)}')
  ./sdpd3cu heat 3 $rows 1 0 1 0 0.01 $tmax 20 1e-3 -1 > heat.$l
  tmax=$(awk -v l=$l 'BEGIN {k = 2 * 3.14159265 / (3 * l); print int(4 / (k * k / 3) / 0.01)}')
  for kappa in 0 2; do
    ./sdpd3cu sound 3 $rows 0.5 0 $kappa 0 0.01 $tmax 10 1e-3 -1 > sound.$kappa.$l
  done
done
for kB in 0.01 0.05; do
  ./sdpd3cu rest 3 60 1 0 1 $kB 0.01 20000 100 0 -1 > rest.$kB
done
