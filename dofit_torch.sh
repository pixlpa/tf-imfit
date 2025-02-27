SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/final/*.png
time python tf-imfit/torch/torch-imfit-nscale.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 800 --iter-multiple 2 --output-dir results/final/ --size 512 --num-gabors 512 --rescales 3 --global-lr 0.01 --mutation-strength 0.001 --gamma 0.998

ffmpeg -framerate 30 -i results/final/result_%04d.png -c:v libx264 -pix_fmt yuv420p -y results/$NAME.mp4
mv results/final/saved_weights.txt results/$NAME.txt
mv results/final/final_result.png results/$NAME.png
cp results/$NAME.mp4 /content/drive/My\ Drive/HM/videos/
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/
cp results/$NAME.png /content/drive/My\ Drive/HM/outputs/