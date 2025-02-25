SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/final/*.png
time python tf-imfit/torch/torch-imfit-nscale.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 4000 --output-dir results/lores/ --size 256 --num-gabors 256 --rescales 2 --global-lr 0.009 --mutation-strength 0.005 --gamma 1 --sobel 0.01

ffmpeg -framerate 30 -i results/final/result_%04d.png -c:v libx264 -pix_fmt yuv420p results/$NAME.mp4
mv results/final/saved_weights.txt results/$NAME.txt
mv results/final/final_result.png results/$NAME.png
cp results/$NAME.mp4 /content/drive/My\ Drive/HM/results/
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/
cp results/$NAME.png /content/drive/My\ Drive/HM/outputs/