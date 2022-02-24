#!/bin/sh

apt-get update
apt-get install -y binutils libproj-dev gdal-bin

# service account setup
gcloud auth activate-service-account sidewalk-service-account@root-grammar-308418.iam.gserviceaccount.com --key-file=root-grammar-308418-74fcad9b6560.json --project=root-grammar-308418

city_name=$1
tile_size=$2
tile_stride=$3
model_name=$4
model_trained_multi_gpu=$5
tile_resize_dim=$6

tile_start_idx=$7
tile_end_idx=$8

feature_type=$9

mkdir -p datasets/"${city_name}"/VRT/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
mkdir -p datasets/"${city_name}"/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
mkdir -p datasets/"${city_name}"/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
mkdir -p datasets/inference_masks/"${feature_type}"/"${city_name}"/"${tile_size}_$((${tile_size}-${tile_stride}))"

gsutil ls gs://earth-observation-bucket/datasets/${city_name}/VRT/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"/*.jp2 > inference_files.txt
gsutil ls gs://earth-observation-bucket/datasets/"${city_name}"/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"/*.jp2 > building_files.txt
gsutil ls gs://earth-observation-bucket/datasets/"${city_name}"/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"/*.jp2 > road_files.txt

# reads each line of "inference_file.txt" and downloads the file to the newly made folder
counter=0
while read i; do
 if [ $counter -ge $tile_start_idx ] && [ $counter -le $tile_end_idx ]; then
  echo "${i}" >> image_subset_tiles.txt
 fi
 counter=$( expr $counter + 1 )
done <inference_files.txt

# reads each line of "building_files.txt" and downloads the file to the newly made folder
counter=0
while read i; do
 if [ $counter -ge $tile_start_idx ] && [ $counter -le $tile_end_idx ]; then
  echo "${i}" >> building_subset_tiles.txt
 fi
 counter=$( expr $counter + 1 )
done <building_files.txt

# reads each line of "road_files.txt" and downloads the file to the newly made folder
counter=0
while read i; do
 if [ $counter -ge $tile_start_idx ] && [ $counter -le $tile_end_idx ]; then
  echo "${i}" >> road_subset_tiles.txt
 fi
 counter=$( expr $counter + 1 )
done <road_files.txt

cat image_subset_tiles.txt | gsutil -m cp -I datasets/"${city_name}"/VRT/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
cat building_subset_tiles.txt | gsutil -m cp -I datasets/"${city_name}"/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
cat road_subset_tiles.txt | gsutil -m cp -I datasets/"${city_name}"/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"


gsutil -m cp -r gs://earth-observation-bucket/models/"${feature_type}"/${model_name} ./

chmod a+rwx tiff2jp2.sh

echo "=================="
date
echo "=================="

python run_inference.py --path_to_tiles ./datasets/"${city_name}"/VRT/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" \
                        --path_to_roads ./datasets/"${city_name}"/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" \
                        --path_to_buildings ./datasets/"${city_name}"/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" \
                        --path_to_model ./"${model_name}"/model_best.pth.tar \
                        --path_to_save_mask ./datasets/inference_masks/"${feature_type}"/"${city_name}"/"${tile_size}_$((${tile_size}-${tile_stride}))" \
                        --model_trained_multi_gpu "${model_trained_multi_gpu}" \
                        --tile_resize_dim "${tile_resize_dim}"

echo "=================="
date
echo "=================="

gsutil -m cp -r datasets/inference_masks/"${feature_type}"/"${city_name}"/"${tile_size}_$((${tile_size}-${tile_stride}))" gs://earth-observation-bucket/datasets/inference_masks/"${feature_type}"/"${city_name}"/

sleep 60
