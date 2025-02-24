SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/final/*.png
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 5000 \
    --output-dir results/final/ --size 256 --num-gabors 256 \
    --global-lr 0.006 --mutation-strength 0.0002 --gamma 0.99

mv results/final/saved_weights.txt results/$NAME.txt
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/