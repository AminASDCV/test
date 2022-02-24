from cluster_utils import load_kube_config, create_gdal_pod, create_gpu_inference_pod, get_pod_phase, node_count_in_pool
import os
import sys
import time
import argparse

def run_inference(args):
	"""
	Steps:
	1. Run Gdal pod for the city
	2. Wait for gdal pod to complete successfully
	3. Read  number of tiles from bucket in inference folder
	4. Calculate start and end index for each go inference container as <NUM_TILES> / 4
	5. Spin 1 pod with 4 gpu inference containers per city
	6. Wait for pods to complete successfuly
	7. Exit 
	"""
	load_kube_config()
	
	print("Step 1")
	gdal_pod_name = create_gdal_pod(args.city_name, args.tile_size, args.tile_stride)
	
	print("Step 2")
	time.sleep(30) #wait buffer for pod creation
	unk_cnt = 0

	# time logging for gdal tasks
	gdal_running_start = 0
	gdal_total = 0

	status = get_pod_phase(gdal_pod_name)
	
	while status != "Succeeded":
		status = get_pod_phase(gdal_pod_name)
		gdal_total += 1

		if status == "Failed":
			print(f'Inference job failed for {args.city_name}')
			return
		elif status == "Unknown":
			unk_cnt += 1
			if unk_cnt > 6:
				print(f'Inference job failed for {args.city_name}')
				return
		elif status == "Running":
			gdal_running_start += 1
		
		time.sleep(10)

	print("Step 3")
	temp = args.tile_size - args.tile_stride
	command = f"gsutil ls gs://earth-observation-bucket/datasets/{args.city_name}/VRT/inference/{args.tile_size}_{temp} > tile_list_{args.city_name}.txt"
	os.system(command)
	
	print("Step 4")
	#MAX_NODES = 10
	num_tiles = 0
	with open("tile_list_"+args.city_name+".txt", 'r') as tfile:
		lines = tfile.readlines()
		num_tiles = len(lines)
	if num_tiles==0:
		print(f'Inference job failed for {args.city_name}. Number of tiles could not be read from bucket')
		return
	num_gpu_nodes = node_count_in_pool('pool-gpu')
	num_tiles_container = (num_tiles//(40 * args.request_gpu_nodes) ) + 1

	print("Step 5")
	#print("scaling up gpu nodes")
	#command = f"gcloud container clusters resize sidewalk-cluster --node-pool pool-gpu --zone us-central1-a --num-nodes {args.request_gpu_nodes} --quiet"
	#os.system(command)
	#print("gpu nodes scaled")
	
	gpu_pod_dict = {}
	start_idx = 0
	end_idx = min(num_tiles-1,start_idx + num_tiles_container-1)
	pod_number = 0
	while start_idx < end_idx:
		try:
			pod_name = create_gpu_inference_pod(args.city_name, args.tile_size, args.tile_stride, args.model_name, 
				args.model_trained_multi_gpu, args.tile_resize_dim, start_idx, end_idx, pod_number)
			gpu_pod_dict[pod_name] = "Unknown"

			start_idx += num_tiles_container
			end_idx = min(num_tiles-1,start_idx + num_tiles_container-1)
			pod_number += 1
		except:
			print(f'Inference job failed for {args.city_name}. GPU pod creation failed')
			return
	print("Step 6")
	# time logging for gpu task 
	
	gpu_running_start = 0
	gpu_total = 0
	gpu_is_running = False

	job_finished=False
	loop_cnt = 0
	
	while not job_finished:
		if loop_cnt > 6000:
			print(f'Inference job failed for {args.city_name}. pods did not succeed after 10 hr')
			return
		
		pods_succeeded = 0
		
		for pod_name in gpu_pod_dict.keys():
			status = get_pod_phase(pod_name)
			
			if status == "Failed":
				print(f'Inference job failed for {args.city_name}. {pod_name} failed')
				return
			elif status == "Succeeded":
				pods_succeeded += 1
			elif status == "Running":
				gpu_is_running = True
			
			gpu_pod_dict[pod_name] = status

		if pods_succeeded == len(gpu_pod_dict.keys()):
			job_finished = True
		else:
			if gpu_is_running:
				gpu_running_start += 1
			gpu_total += 1
			time.sleep(10)
		
		loop_cnt += 1
	
	print(f'Inference job finished for {args.city_name}')

	# writing the time variables to a file
	with open('job_timing.txt', "a") as f:
		#f.write(f'GDAL pod total time for {args.city_name} = {gdal_total * 10} seconds')
		#f.write(f'GDAL pod running time for {args.city_name} = {gdal_running_start * 10} seconds')
		
		f.write(f'GPU pod total time for {args.city_name} = {gpu_total * 10} seconds')
		f.write(f'GPU pod running time for {args.city_name} = {gpu_running_start * 10} seconds')

	return


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Arguments for gdal container pod')
	parser.add_argument('--city_name',type=str,required=True,help="city to be processed")
	parser.add_argument('--tile_size',type=int,required=False,default=1250,help='tile size')
	parser.add_argument('--tile_stride',type=int,required=False,default=1100,help='tile stride')
	parser.add_argument('--model_name',type=str,required=True,help='name of model')
	parser.add_argument('--model_trained_multi_gpu',type=bool,required=True,help='multi gpu model?')
	parser.add_argument('--tile_resize_dim',type=int,required=False, default=768, help='tile resize dimention')
	parser.add_argument('--request_gpu_nodes',type=int,required=False, default=1, help='number of gpu nodes')

	args =parser.parse_args()

	run_inference(args)
