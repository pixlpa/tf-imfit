mkdir -p results results/lores results/midres results/hires results/final

# 2m14s to get to ~0.00014
rm results/lores/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 64 -T 256 -o results/weights_lores.txt -S -x results/lores/out

# Note we do full updates more often, train/replace 2 models at a time, fuzz a bit more
# 8m1s to get to 0.00034
rm results/midres/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 96 -T 384 -i results/weights_lores.txt -o results/weights_midres.txt -S -x results/midres/out

#15m15s to 0.000376
rm results/hires/*png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 128 -T 512 -a0.03 -R 0.0002 -i results/weights_midres.txt -o results/weights_hires.txt -S -x results/hires/out

# 1m54s to 0.00082
rm results/final/*png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 256 -t 0:00:01 -R 0.0002 -i results/weights_hires.txt -o results/weights_final.txt -S -x results/final/out
