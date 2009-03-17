set term svg

set logscale xy
set logscale y2

set autoscale y2
set ytics nomirror
set y2tics

set xlabel "# of Memory Regions"
set ylabel "Slow-down"
set y2label "Slow-down Slope"
set key outside top center horizontal

set style fill solid 0.5 border 0.5
set boxwidth 0.8
set style line 1 lt 1 lc rgb "red" lw 2
set style line 2 lt 2 lc rgb "red"
set style line 3 lt 1 lc rgb "blue"

set output "signal-lines.svg"
set style fill solid 0.2 border -1
plot\
'signal.data' u :($3):($4) title "Slow-Down" w filledcurves,\
'signal.data' u ($1):($2) notitle w lines ls 1,\
'signal.data' u ($1):($3) notitle w lines ls 2,\
'signal.data' u ($1):($4) notitle w lines ls 2,\
'signal.data' u ($1):($5) title "Slope" w lines ls 3 axes x1y2
