SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/final/*.png
time python tf-imfit/torch/torch-imfit3a.py $SOURCEIMG --iterations 3000 \
    --single-iterations 5 --output-dir results/final/ --size 256 --num-gabors 256 \
    --global-lr 0.009 --mutation-strength 0.0

mv results/final/saved_weights.txt results/$NAME.txt
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/