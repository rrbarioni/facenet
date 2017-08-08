#!/usr/bin/gnuplot

set term png
set size .75,1
set output "lfw_restricted_strict.png"
set xtics .1
set ytics .1
set grid
set size ratio -1
set ylabel "true positive rate"
set xlabel "false positive rate"
set title "Image-Restricted, No Outside Data" font "giant"
set key right bottom
plot "tpr_fpr_alignedLfw_list.txt" using 2:1 with lines title "FaceNet, original"