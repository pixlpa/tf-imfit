SOURCEIMG=$1
NAME=$(basename $SOURCEIMG .png)

echo $SOURCEIMG $NAME $NAME.txt

rm -f results/*.png
rm -f results/lores/*.png
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 4000 --output-dir results/lores/ --size 128 --num-gabors 256 --global-lr 0.009 --mutation-strength 0.0 --gamma 0.9995
rm -f results/hires/*.png
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 3000 --output-dir results/hires/ --init results/lores/saved_model.pth --size 256 --init-size 128 --num-gabors 256 --global-lr 0.005 --mutation-strength 0.005 --gamma 0.99999
rm -f results/final/*.png
time python tf-imfit/torch/torch-imfit5.py $SOURCEIMG --weight source_weights/$NAME-wt.png --iterations 1500 --output-dir results/final/ --init results/hires/saved_model.pth --size 512 --init-size 256 --num-gabors 256 --global-lr 0.004 --local-lr 0.001 --mutation-strength 0.0 --gamma 0.99999

mv results/final/saved_weights.txt results/$NAME.txt
mv results/final/final_result.png results/$NAME.png
cp results/$NAME.txt /content/drive/My\ Drive/HM/results/
cp results/$NAME.png /content/drive/My\ Drive/HM/outputs/