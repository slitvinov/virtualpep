#! /bin/sh

set -eu

eta=10
etas=0.2
for dim in 2 3; do
    if [ $dim = 2 ]; then row=0.9306049; c0=1.4142136; else row=0.7937005; c0=1.2909944; fi
    for h in 3 4; do
	for lh in 1 1.25 1.5 2 2.5 3 4 6 8 12 16; do
	    rows=$(awk -v l=$lh -v h=$h -v r=$row 'BEGIN { printf "%d", l * h / r + 0.5 }')
	    k=$(awk -v n=$rows -v r=$row 'BEGIN { print 2 * 3.141592653589793 / (n * r) }')
	    dt=$(awk -v k=$k -v e=$eta 'BEGIN { t = 0.05 / (e * k * k); print (t < 0.01 ? t : 0.01) }')
	    steps=$(awk -v k=$k -v e=$eta -v dt=$dt 'BEGIN { printf "%d", 2.5 / (e * k * k) / dt }')
	    ./sdpd$dim shear $h $rows $eta 0 0 0 $dt $steps 1 |
		awk -v e=$eta 'NR == 1 { lh = $3; k = $5; next }
		    NR == 2 { a0 = $2 }
		    $2 > a0 * exp(-2) { n++; x = $1; y = log($2); sx += x; sy += y; sxx += x * x; sxy += x * y }
		    END { s = (n * sxy - sx * sy) / (n * sxx - sx * sx); print lh, -s / (e * k * k) }'
	done > nu$dim.$h
	for lh in 1 1.25 1.5 2 2.5 3 4 6 8 12 16; do
	    rows=$(awk -v l=$lh -v h=$h -v r=$row 'BEGIN { printf "%d", l * h / r + 0.5 }')
	    k=$(awk -v n=$rows -v r=$row 'BEGIN { print 2 * 3.141592653589793 / (n * r) }')
	    steps=$(awk -v k=$k -v c=$c0 'BEGIN { printf "%d", 4 * 2 * 3.141592653589793 / (0.5 * c * k) / 0.01 }')
	    ./sdpd$dim sound $h $rows $etas 0 0 0 0.01 $steps 1 |
		awk -v c0=$c0 'NR == 1 { lh = $3; k = $5; next }
		    { if (m > 0 && p * $2 < 0) z[nz++] = t - p * ($1 - t) / ($2 - p); p = $2; t = $1; m++ }
		    END { print lh, 3.141592653589793 * (nz - 1) / (z[nz - 1] - z[0]) / k / c0 }'
	done > c$dim.$h
    done
done
