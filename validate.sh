# checks of sdpd3cu and sdpd3b; run in an empty directory next to the sources
set -e
src=${1:-..}
nvcc -O3 -arch=sm_90 -fmad=false $src/sdpd3.cu -o sdpd3cu
nvcc -O3 -arch=sm_90 -fmad=false $src/sdpd3b.cu -o sdpd3b
nvcc -O3 -arch=sm_90 -fmad=false -DLUCY $src/sdpd3.cu -o sdpd3cul
nvcc -O3 -arch=sm_90 -fmad=false -DLUCY $src/sdpd3b.cu -o sdpd3bl
gcc -O3 -fopenmp $src/sdpd3.c -lm -o sdpd3

same() { cmp -s $1 $2 && echo "ok   $3" || { echo "FAIL $3"; diff $1 $2 | head -4; }; }

echo "1. repeatability"
./sdpd3cu shear 3 32 1 0.5 1 1e-3 0.02 300 50 > r1 2>/dev/null
./sdpd3cu shear 3 32 1 0.5 1 1e-3 0.02 300 50 > r2 2>/dev/null
same r1 r2 "sdpd3cu twice"

echo "2. one box"
echo "b1 shear 3 32 1 0.5 1 1e-3 0.02 300 50" | ./sdpd3b 2>/dev/null
same r1 b1 "sdpd3b one box = sdpd3cu"

echo "3. mixed batch"
cat > list <<END
m1 shear 3 32 1 0.5 1 1e-3 0.02 300 50
m2 sound 4 24 0.3 0 2 1e-2 0.01 400 40 2e-3 1
m3 heat 3 30 1 0 1 1e-3 0.01 250 25 1e-3 1
m4 rest 2.5 20 2 1 0 5e-2 0.005 350 70
END
./sdpd3b < list 2>/dev/null
./sdpd3cu shear 3 32 1 0.5 1 1e-3 0.02 300 50 > s1 2>/dev/null
./sdpd3cu sound 4 24 0.3 0 2 1e-2 0.01 400 40 2e-3 -1 > s2 2>/dev/null
./sdpd3cu heat 3 30 1 0 1 1e-3 0.01 250 25 1e-3 -1 > s3 2>/dev/null
./sdpd3cu rest 2.5 20 2 1 0 5e-2 0.005 350 70 > s4 2>/dev/null
for k in 1 2 3 4; do same m$k s$k "batch box $k = single"; done
echo "m1 shear 3 32 1 0.5 1 1e-3 0.02 300 50 1e-3 0 7" | ./sdpd3b 2>/dev/null
cmp -s m1 r1 && echo "FAIL seed 7 gives the same noise" || echo "ok   seed 7 changes the noise"

echo "4. CPU and GPU without noise"
./sdpd3 shear 3 32 1 0.5 1 0 0.02 300 50 > c0
./sdpd3cu shear 3 32 1 0.5 1 0 0.02 300 50 > g0 2>/dev/null
paste c0 g0 | awk 'NR > 1 {a = $2 - $8; e = $3 - $9; if (a < 0) a = -a; if (e < 0) e = -e;
  if (a > ma) ma = a; if (e > me) me = e} END {printf "     max |amplitude difference| %.3g, max |energy difference| %.3g\n", ma, me}'
gcc -O3 -fopenmp -DLUCY $src/sdpd3.c -lm -o sdpd3l && ./sdpd3l shear 3 32 1 0.5 1 0 0.02 300 50 > cw
./sdpd3cul shear 3 32 1 0.5 1 0 0.02 300 50 > gw 2>/dev/null
paste cw gw | awk 'NR > 1 {a = $2 - $8; if (a < 0) a = -a; if (a > ma) ma = a} END {printf "     Lucy: max |amplitude difference| %.3g\n", ma}'

echo "5. conservation"
./sdpd3cu rest 3 40 1 0.5 1 1e-2 0.01 5000 500 > c5 2>/dev/null
awk 'NR == 2 {e0 = $3} NR > 1 {d = ($3 - e0) / e0; if (d < 0) d = -d; if (d > m) m = d;
  for (k = 8; k <= 10; k++) {p = $k < 0 ? -$k : $k; if (p > mp) mp = p}}
  END {printf "     max relative energy change %.3g, max |total momentum| %.3g\n", m, mp}' c5
for k in 1 2 4; do
  ./sdpd3cu sound 3 20 0.5 0 1 0 $(awk -v k=$k 'BEGIN {print 0.04 / k}') $((100 * k)) $k > e$k 2>/dev/null
  awk -v k=$k 'NR == 2 {e0 = $3} NR > 1 {d = $3 - e0; if (d < 0) d = -d; if (d > m) m = d}
    END {printf "     dt = %.3g: max |energy change| %.3g\n", 0.04 / k, m}' e$k
done
echo "w1 rest 3 40 1 0.5 1 1e-2 0.01 5000 500" | ./sdpd3bl 2>/dev/null
./sdpd3cul rest 3 40 1 0.5 1 1e-2 0.01 5000 500 > w2 2>/dev/null
same w1 w2 "Lucy batch = single"

echo "6. compute-sanitizer"
head -2 list > small
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool $tool --error-exitcode 1 ./sdpd3b < small > /dev/null 2>san.$tool &&
    echo "ok   $tool sdpd3b" || { echo "FAIL $tool sdpd3b"; tail -5 san.$tool; }
  compute-sanitizer --tool $tool --error-exitcode 1 ./sdpd3cu sound 3 20 0.5 0 1 1e-3 0.01 40 10 > /dev/null 2>san.$tool &&
    echo "ok   $tool sdpd3cu" || { echo "FAIL $tool sdpd3cu"; tail -5 san.$tool; }
done
