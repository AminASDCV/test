import os,sys
import argparse
from tqdm import tqdm
import concurrent.futures

import rasterio as rio
from rasterio import features
import geopandas as gpd


############## Setup argparse
parser = argparse.ArgumentParser()
    
parser.add_argument(
    '--path_tiles', type=str, required=True, help='path to input tiles')
parser.add_argument(
    '--path_shpfile', type=str, required=True, help='path to polyline shape file')
parser.add_argument(
    '--path_output', type=str, required=True, default='out_tiles/', help='path output')
parser.add_argument(
    '--sw_width', type=float, required=False, default=4, help='total line width wanted')

args = parser.parse_args()


############## Parameters required
path_tiles = args.path_tiles #"/mnt/mount-point-directory/geomate_sidewalk_detection/to_shaeri_for_preprocessing_code/sample_tiles/" 
path_shpfile = args.path_shpfile #"/mnt/mount-point-directory/geomate_sidewalk_detection/to_shaeri_for_preprocessing_code/kw_test.geojson"
buffer_dist = (args.sw_width)/2
path_output = args.path_output

############## Create outpath if does not exist
if not os.path.isdir(path_output):
    os.makedirs(path_output)
    
    
############## Reading tiles from directory 
tilenames = []

for file in os.listdir(path_tiles):
    if file.endswith('.jp2'):
        tilenames.append(file)
        
print(f'Processing ===> {len(tilenames)} files.')


############## Reading the first tile to get the coordination system of dataset
template = rio.open(path_tiles + tilenames[0])


############## Reading shapefile and transforming its coordination system
gs = gpd.GeoSeries.from_file(path_shpfile)
gs = gs.to_crs(template.crs).buffer(buffer_dist)

shapes = [[feature,255] for feature in gs]


############## Function to process a single tile
def sidewalk_mask_rasterize(tilename):
    tile = rio.open(path_tiles + tilename)
    meta = tile.meta.copy()
    meta['count'] = 1
    
    with rio.open(path_output + tilename, 'w+', **meta) as out:
        out_arr = out.read(1)
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
        
    
if __name__ == "__main__":
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(sidewalk_mask_rasterize, tilenames)))