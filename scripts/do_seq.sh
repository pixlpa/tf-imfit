rm -f results/torch/*.png
rm -f results/*.txt
echo Training Initial Frame
python tf-imfit/torch/torch-imfit-nscale.py source_images/img0001.png --weight source_weights/img0001-wt.png --iterations 300 --rescales 2 --output-dir results/torch/ --size 256 --num-gabors 512 --global-lr 0.01 --mutation-strength 0.0001 --gamma 0.997
mv results/torch/saved_weights.txt results/img0001.txt
mv results/torch/final_result.png results/img0001.png

echo Finished initial frame

PREV=results/img0001.txt
rm -f results/torch/*.png
count=0
for img in source_images/*.png
do
  NAME=$(basename $img .png)
  echo Training Frame $NAME
  IN=$img
  WT=source_weights/$NAME-wt.png
  if (( count % 15 == 0)) ; then
    echo Training Key Frame
    python -u tf-imfit/torch/torch-imfit-nscale.py $IN --weight $WT --iterations 300 --rescales 2 --output-dir results/torch/ --size 256 --num-gabors 512 --global-lr 0.01 --mutation-strength 0.0 --gamma 0.997
  else
    echo Training Standard Frame
    python -u tf-imfit/torch/torch-imfit-nscale.py $IN --weight $WT --iterations 300 --rescales 1 --output-dir results/torch/ --init $PREV --size 256 --num-gabors 512 --global-lr 0.01 --mutation-strength 0.0 --gamma 0.997
  fi
  ((count++))
  mv results/torch/saved_weights.txt results/$NAME.txt
  mv results/torch/final_result.png results/images/$NAME.png
  cp results/$NAME.txt /content/drive/My\ Drive/HM/vid-out/
  PREV=results/$NAME.txt
  echo completed $NAME
done