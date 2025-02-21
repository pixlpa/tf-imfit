SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/lores/*.png && \
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG \
    --iterations 1000 --single-iterations 5 --output-dir results/lores/ \
    --size 128 --num-gabors 256 --phase-split 0.6 --global-lr 0.02 \
    --local-lr 0.005 --mutation-strength 0.0002

rm -f results/final/*.png && \
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG \
    --iterations 2000 --single-iterations 5 --output-dir results/final/ \
    --size 256 --num-gabors 256 --phase-split 0.6 --global-lr 0.02 \
    --local-lr 0.005 --mutation-strength 0.0002 --init results/lores/saved_model.pth \
    --init-size 128

mv results/final/saved_weights.txt results/$NAME.txt
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/