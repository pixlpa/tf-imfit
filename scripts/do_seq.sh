rm -f results/torch/*.png
for i in {1...23} 
do
    initname=$(printf "results/img%04d.txt" $(($i-1)))
    OUT=$(printf "results/img%04d" $i)
    python tf-imfit/torch/torch-imfit-nscale.py source_vid/face-moves$i.png --iterations 200 --rescales 3 --output-dir results/torch/ --init results/img0001.txt --size 512 --num-gabors 256 --global-lr 0.009 --mutation-strength 0.0 --gamma 0.997
    mv results/torch/saved_weights.txt $OUT.txt;mv results/torch/final_result.png $OUT.png
done
echo "completed!!!"