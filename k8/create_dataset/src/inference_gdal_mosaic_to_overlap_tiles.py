import os
import glob
import numpy as np
from rasterio import features
import rasterio as rio
import concurrent.futures
import tqdm
import argparse
import geopandas as gpd

parser = argparse.ArgumentParser(description='Provide city name to process.')
parser.add_argument('--city', type=str, required=True, help='name of inference city')
parser.add_argument('--tile_size', type=int, required=False, help='size of individual tile')
parser.add_argument('--tile_stride', type=int, required=False, help='stride of individual tile overlap')
parser.add_argument('--building_fp_shapefile', type=str, required=True, help='path to building footprint shapefile folder')
parser.add_argument('--road_shapefile', type=str, required=True, help='path to road shapefile folder')
args = parser.parse_args()

if args.tile_size:
    X_TILE = args.tile_size # tile width
    Y_TILE = args.tile_size # tile height
else:
    X_TILE = 1250			# tile width
    Y_TILE = 1250			# tile height

if args.tile_stride:
    X_INC = args.tile_stride # tile width overlap, if X_TILE = X_INC then no overlap
    Y_INC = args.tile_stride # tile height overlap, if Y_TILE = Y_INC then no overlap
else:
    X_INC = 1100			# tile width overlap, if X_TILE = X_INC then no overlap
    Y_INC = 1100			# tile height overlap, if Y_TILE = Y_INC then no overlap

# Read and Save directories
IMAGE_NAMES = glob.glob(f'./datasets/{args.city}/VRT/city_images/{args.city}.vrt')
SAVE_DIRECTORY_IMAGE_TILES = f'./datasets/{args.city}/VRT/inference/{X_TILE}_{X_TILE-X_INC}'
SAVE_DIRECTORY_BUILDING_FP = f'./datasets/{args.city}/mask_building_fp/inference/{X_TILE}_{X_TILE-X_INC}'
SAVE_DIRECTORY_ROAD = f'./datasets/{args.city}/mask_road/inference/{X_TILE}_{X_TILE-X_INC}'

image_names_list = []
xy_clip_points = []

#Read shapefiles:
shpfile_building = None
shpfile_road = None
for file in os.listdir(args.building_fp_shapefile):
    if "json" in file.split(".")[1].lower() or file.split(".")[1].lower() == "shp":
        shpfile_building = os.path.join(args.building_fp_shapefile,file)
        break
for file in os.listdir(args.road_shapefile):
    if "json" in file.split(".")[1].lower() or file.split(".")[1].lower() == "shp":
        shpfile_road = os.path.join(args.road_shapefile,file)
        break

if shpfile_road is None or shpfile_building is None:
    print('No shapefiles provided')
    exit(1)


print(shpfile_building)
print(shpfile_road)

gs_building_glob = gpd.GeoSeries.from_file(shpfile_building)
gs_road_glob = gpd.GeoSeries.from_file(shpfile_road)

    
def create_processing_lists():

    for image_name in IMAGE_NAMES:
    	image = rio.open(image_name)

    	for x in np.arange(0, image.meta['width'], X_INC).tolist():
    	    for y in np.arange(0, image.meta['height'], Y_INC).tolist():
    	        image_names_list.append(image_name)
    	        xy_clip_points.append([x,y])

def mosaic_to_overlap(img_name, xy_clip):
    
    # implementing for img
    _outfile_img_name = img_name.split('/')[-1].split('.')[0]
    outfile_img_name = _outfile_img_name + '_' + str(xy_clip[0]) + '_' + str(xy_clip[1]) + '.jp2'
    
    print('Processing vrt_clip ---> {}'.format(outfile_img_name))
    
    infile_img_path = img_name
    outfile_img_path = os.path.join(SAVE_DIRECTORY_IMAGE_TILES, outfile_img_name)
    
    command = 'gdal_translate -of JP2OpenJPEG -b 1 -b 2 -b 3 -srcwin {} {} {} {} {} {}'.format(xy_clip[0], xy_clip[1],
    											X_TILE, Y_TILE,
    											infile_img_path, outfile_img_path)
                                                                           
    os.system(command)

    mask_rasterize(outfile_img_path)


def mask_rasterize(path_tile,buffer_dist=2):
    _out_tile = path_tile.split('/')[-1]
    out_tile_building = os.path.join(SAVE_DIRECTORY_BUILDING_FP,_out_tile)
    out_tile_road = os.path.join(SAVE_DIRECTORY_ROAD,_out_tile)

    ############## Reading tile
    tile = rio.open(path_tile)
    
    ############## projecting shapefile and transforming its coordination system
    gs_building = gs_building_glob.to_crs(tile.crs)

    gs_road = gs_road_glob.to_crs(tile.crs).buffer(buffer_dist)

    ############## Building

    shapes = [[feature,255] for feature in gs_building]

    ############## Function to process a single tile
    meta = tile.meta.copy()
    meta['count'] = 1
    
    with rio.open(out_tile_building, 'w+', **meta) as out:
      out_arr = out.read(1)
      burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
      out.write_band(1, burned)
    #mask_arr = np.zeros(cv2.imread(path_tile, 0).shape)
    #mask_arr = features.rasterize(shapes=shapes, fill=0, out=mask_arr, transform=tile.transform)
    #return mask_arr

    ######### Road
    shapes = [[feature,255] for feature in gs_road]

    ############## Function to process a single tile
    meta = tile.meta.copy()
    meta['count'] = 1
    
    with rio.open(out_tile_road, 'w+', **meta) as out:
      out_arr = out.read(1)
      burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
      out.write_band(1, burned)


    
def main():

    if not os.path.isdir(SAVE_DIRECTORY_IMAGE_TILES):
        os.makedirs(SAVE_DIRECTORY_IMAGE_TILES)
        print('Created directory --> {}',format(SAVE_DIRECTORY_IMAGE_TILES))

    create_processing_lists()	# create lists to pass to ProcessPoolExecutor
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tqdm.tqdm(executor.map(mosaic_to_overlap, image_names_list, xy_clip_points))


if __name__ == "__main__":
    main()
