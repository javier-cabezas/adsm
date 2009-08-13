set term svg

set logscale x

set autoscale y2
set ytics nomirror
set y2tics

set xlabel "# of Memory Regions"
set ylabel "Slow-down"
set y2label "Slow-down Slope"
set key outside top center horizontal

set style fill solid 0.5 border 0.5
set boxwidth 0.8
set style line 1 lt 1 lc rgb "red" 
set style line 2 lt 1 lc rgb "blue"

set output "signal-lines.svg"
set style fill solid 0.2 border -1
plot\
'signal.data' u ($1):($2) title "Slow-Down" w lines ls 1,\
'signal.data' u ($1):($5) title "Slope" w lines ls 2 axes x1y2
