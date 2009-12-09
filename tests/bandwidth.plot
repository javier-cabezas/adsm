set term svg

set xlabel "Size in bytes"
set ylabel "Gbps"
set key outside top center horizontal
set xtics rotate

set style fill solid 0.5 border 0.5
set boxwidth 0.8
set style line 1 lt 1 lc rgb "red" lw 2
set style line 2 lt 1 lc rgb "blue" lw 2
set style line 3 lt 2 lc rgb "red"
set style line 4 lt 2 lc rgb "blue"

set output "bandwidth-bar.svg"
plot\
'bandwidth.data' u ($4):xtic(1) w boxes ls 1 title "GPU Input Bandwidth",\
'bandwidth.data' u ($5):xtic(1) w boxes ls 2 title "GPU Output Bandwidth",\
'bandwidth.data' u :($4):($6):($8):xtic(1) notitle w yerrorbars 1,\
'bandwidth.data' u :($5):($7):($9):xtic(1) notitle w yerrorbars 2

set output "bandwidth-lines.svg"
set style fill solid 0.2 border -1
plot\
'bandwidth.data' u :($6):($8):xtic(1) w filledcurves title "GPU Input Bandwidth",\
'bandwidth.data' u :($7):($9):xtic(1) w filledcurves title "GPU Output Bandwidth",\
'bandwidth.data' u ($4):xtic(1) notitle w lines ls 1,\
'bandwidth.data' u ($6):xtic(1) notitle w lines ls 3,\
'bandwidth.data' u ($8):xtic(1) notitle w lines ls 3,\
'bandwidth.data' u ($5):xtic(1) notitle w lines ls 2,\
'bandwidth.data' u ($7):xtic(1) notitle w lines ls 4,\
'bandwidth.data' u ($9):xtic(1) notitle w lines ls 4
