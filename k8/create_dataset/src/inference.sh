#!/bin/sh

# service account setup
gcloud auth activate-service-account sidewalk-service-account@root-grammar-308418.iam.gserviceaccount.com --key-file=root-grammar-308418-74fcad9b6560.json --project=root-grammar-308418

city_name=$1
tile_size=$2
tile_stride=$3

mkdir -p datasets/"${city_name}"/VRT
mkdir -p datasets/"${city_name}"/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
mkdir -p datasets/"${city_name}"/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))"
mkdir -p datasets/"${city_name}"/annotations

gsutil -m cp -r gs://earth-observation-bucket/datasets/"${city_name}"/VRT/city_images datasets/"${city_name}"/VRT/
#Copy shape files of builing fp and  road centerlines
gsutil cp -r gs://earth-observation-bucket/datasets/"${city_name}"/annotations/buildings datasets/"${city_name}"/annotations/
gsutil cp -r gs://earth-observation-bucket/datasets/"${city_name}"/annotations/roads datasets/"${city_name}"/annotations/

# create VRT of tiles, only accepts .JP2 and .TIF
cd datasets/"${city_name}"/VRT/city_images/
ls | grep .jp2 > tiles_list.txt
ls | grep .tif >> tiles_list.txt

gdalbuildvrt -input_file_list tiles_list.txt "${city_name}".vrt

cd /home/src

# create tiles of {tile_size} with {tile_stride}
python inference_gdal_mosaic_to_overlap_tiles.py --city ${city_name} --tile_size ${tile_size} --tile_stride ${tile_stride} --building_fp_shapefile /home/src/datasets/"${city_name}"/annotations/buildings  --road_shapefile /home/src/datasets/"${city_name}"/annotations/roads

# syncing with the bucket
gsutil -m cp -r datasets/${city_name}/VRT/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" gs://earth-observation-bucket/datasets/${city_name}/VRT/inference/

gsutil -m cp -r datasets/${city_name}/mask_building_fp/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" gs://earth-observation-bucket/datasets/"${city_name}"/mask_building_fp/inference/
gsutil -m cp -r datasets/${city_name}/mask_road/inference/"${tile_size}_$((${tile_size}-${tile_stride}))" gs://earth-observation-bucket/datasets/"${city_name}"/mask_road/inference/
