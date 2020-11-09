SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME-wt.png $NAME.txt

rm -f results/lores/*.png && \
time python tf-imfit/rgb-imfit.py $SOURCEIMG -w source_weights/$NAME-wt.png \
     -s 64 -T 256 -a 0.01 \
     -o results/weights_lores.txt \
     -p 512 -S -x results/lores/out -n 128

rm -f results/lores/*.png && \
time python tf-imfit/rgb-imfit.py $SOURCEIMG -w source_weights/$NAME-wt.png \
     -s 96 -T 256 -a 0.04 \
     -i results/weights_lores.txt -o results/weights_midres.txt \
     -p 512 -S -x results/midres/out

rm -f results/hires/*.png && \
time python tf-imfit/rgb-imfit.py $SOURCEIMG -w source_weights/$NAME-wt.png \
     -s 128 -T 128 -a 0.02 -R 0.0002 \
     -i results/weights_midres.txt -o results/weights_hires.txt \
     -p 512 -S -x results/hires/out -n 128

rm -f results/final/*.png && \
time python tf-imfit/rgb-imfit.py $SOURCEIMG -w source_weights/$NAME-wt.png \
    -s 256 -t 0:00:01 -R 0.0002 \
     -i results/weights_hires.txt -o results/weights_final.txt \
     -p 512 -S -x results/final/out -n 128

python tf-imfit/makeparams-rgb.py results/weights_final.txt results/$NAME.txt

cp results/$NAME.txt /content/drive/My\ Drive/Arca/results/