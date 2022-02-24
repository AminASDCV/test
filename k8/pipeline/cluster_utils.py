from kubernetes import client, config
from kubernetes.client.models import (V1Pod, V1VolumeMount, V1ObjectMeta, V1Container, V1PodSpec, 
	V1PersistentVolumeClaim, V1Volume,V1PersistentVolumeClaimVolumeSource, V1PodStatus,V1Toleration)
import os
import sys
import argparse

def load_kube_config():
	config.load_kube_config()

def list_all_pods():
	v1 = client.CoreV1Api()
	print("Listing pods with their IPs:")
	ret = v1.list_pod_for_all_namespaces(watch=False)
	for i in ret.items:
		print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

def get_pod_phase(pod_name):
	v1 = client.CoreV1Api()
	pod_list = v1.list_namespaced_pod("default")
	for pod in pod_list.items:
		if pod.metadata.name == pod_name:
			return pod.status.phase
		#print("%s\t%s\t%s" % (pod.metadata.name, 
        #                  pod.status.phase,
        #                  pod.status.pod_ip))

def node_count_in_pool(pool_name):
	v1 = client.CoreV1Api()
	response = v1.list_node()
	count = 0
	for node in response.items:
		if node.metadata.annotations["node.gke.io/last-applied-node-labels"].split(',')[2].split('=')[1] == pool_name:
			count += 1
	return count

def create_gdal_pod(city_name,tile_size, tile_stride):
	# Configs can be set in Configuration class directly or using helper utility
	v1 = client.CoreV1Api()

	pod = client.V1Pod()
	pod.metadata = client.V1ObjectMeta(name='gdal-pod-'+city_name)

	#requirement = client.V1ResourceRequirements(limits={'cpu':'90','memory':'90G','ephemeral-storage':'90G'}, requests={'cpu':'90','memory':'90G','ephemeral-storage':'90G'})
	requirement = client.V1ResourceRequirements(limits={'cpu':'90','memory':'90G'}, requests={'cpu':'90','memory':'90G'})
	
	#volume_mount = client.V1VolumeMount(name='pv-sidewalk', mount_path="/home/nfs")
	volume_mount = client.V1VolumeMount(name='mystorage', mount_path="/home/src/datasets")

	container=client.V1Container(name='gdal-container-'+city_name, 
		image='gcr.io/root-grammar-308418/driveway:v1',
		resources=requirement, volume_mounts=[volume_mount])
	container.command=['sleep','4800']
	#container.command=['sh','inference.sh']
	#container.args = [city_name,str(tile_size), str(tile_stride)]
	
	#claim =client.V1PersistentVolumeClaimVolumeSource(claim_name='fileserver-claim')
	#volume=client.V1Volume(name='pv-sidewalk',persistent_volume_claim=claim)
	
	empty_dir = client.V1EmptyDirVolumeSource(medium='',size_limit='180G')
	volume=client.V1Volume(name='mystorage', empty_dir=empty_dir)
	
	node_selector = {"cloud.google.com/gke-nodepool":"pool-gdal"}
	tolerations = client.V1Toleration(key='dedicated',operator='Equal',value='experimental',effect='NoSchedule')

	spec = client.V1PodSpec(containers=[container], restart_policy="Never", node_selector=node_selector, volumes=[volume], tolerations=[tolerations])
	pod.spec = spec
	v1.create_namespaced_pod(namespace='default',body=pod)
	print("GDAL Pod deployed")
	return 'gdal-pod-'+city_name

def create_gpu_inference_pod(city_name, tile_size, tile_stride, model_name, model_trained_multi_gpu, 
							tile_resize_dim, tile_start_idx, tile_end_idx, pod_number):
	v1 = client.CoreV1Api()
	pod = client.V1Pod()
	pod.metadata = client.V1ObjectMeta(name='gpu-inference-pod-'+str(pod_number)+'-'+city_name)

	requirement = client.V1ResourceRequirements(limits={'cpu':'12','memory':'32G','nvidia.com/gpu':'1'}, requests={'cpu':'12','memory':'32G','nvidia.com/gpu':'1'})
	
	volume_mount = client.V1VolumeMount(name='dshm', mount_path="/dev/shm")

	container=client.V1Container(name='gpu-inference-container-'+str(pod_number)+'-'+city_name, 
		image='gcr.io/root-grammar-308418/driveway-gpu-inference:latest',
		resources=requirement, volume_mounts=[volume_mount])
	#container.command=['sleep','4800']
	container.command=['sh','inference-gpu.sh']
	container.args = [city_name,str(tile_size), str(tile_stride),model_name, str(model_trained_multi_gpu), 
							str(tile_resize_dim), str(tile_start_idx), str(tile_end_idx)]
	
	#claim =client.V1PersistentVolumeClaimVolumeSource(claim_name='fileserver-claim')
	
	empty_dir = client.V1EmptyDirVolumeSource(medium='Memory',size_limit='4Gi') #set medium='Memory' for RAM, "size_limit" argument in case memory is exceeded 
	volume=client.V1Volume(name='dshm', empty_dir=empty_dir)
	
	node_selector = {"cloud.google.com/gke-nodepool":"pool-gpu"}
	tolerations = [client.V1Toleration(key='nvidia.com/gpu',operator='Equal',value='present',effect='NoSchedule'),client.V1Toleration(key='dedicated',operator='Equal',value='experimental',effect='NoSchedule')]

	spec = client.V1PodSpec(containers=[container], node_selector=node_selector, restart_policy="Never", volumes=[volume], tolerations=tolerations)#host_ipc=True
	pod.spec = spec
	v1.create_namespaced_pod(namespace='default',body=pod)
	print(f"GPU Inference Pod {pod_number} for {city_name} deployed")
	return 'gpu-inference-pod-'+str(pod_number)+'-'+city_name


def create_pre_training_pod(pod_name):
	v1 = client.CoreV1Api()
	pod = client.V1Pod()
	pod.metadata = client.V1ObjectMeta(name = 'gpu-pre-training-pod-' + pod_name)

	# setup computation and volume requirements of the pod
	requirements = client.V1ResourceRequirements(limits = {'cpu':'224','memory':'224G'}, 
												 requests = {'cpu':'218','memory':'218G'})
	volume_mount_1 = client.V1VolumeMount(name='mypv', 
										  mount_path="/mnt/mount-point-directory/datasets2")

	# setup container to run in the pod
	container=client.V1Container(name = 'gpu-train-container-'+ pod_name,
								 image = 'gcr.io/root-grammar-308418/sidewalk-training:v2',
								 resources = requirements,
								 volume_mounts = [volume_mount_1])

	# command to run on the container
	container.command=['sleep', '3600']
	# container.args = [pod_name]

	# create persistentVolumeClaim
	claim = client.V1PersistentVolumeClaimVolumeSource(claim_name = 'pvc-demo')
	volume_1 = client.V1Volume(name = 'mypv', 
							   persistent_volume_claim = claim)

	# select the node to run the pod
	node_selector = {"cloud.google.com/gke-nodepool":"pool-gdal"}

	# create the pod spec
	spec = client.V1PodSpec(containers = [container], 
							node_selector = node_selector, 
							restart_policy="Never", 
							volumes=[volume_1])
	pod.spec = spec

	# create the pod
	v1.create_namespaced_pod(namespace = 'default', body = pod)
	print(f"Preprocessing Training Pod: {pod_name} deployed")
	return 'gpu-pre-training-pod-' + pod_name


def create_gpu_training_pod(experiment_name):
	v1 = client.CoreV1Api()
	pod = client.V1Pod()
	pod.metadata = client.V1ObjectMeta(name='gpu-train-pod-'+experiment_name)

	requirement = client.V1ResourceRequirements(limits={'cpu':'12','memory':'82G','nvidia.com/gpu':'1'}, requests={'cpu':'11','memory':'80G','nvidia.com/gpu':'1'})
	
	volume_mount_1 = client.V1VolumeMount(name='mypv', mount_path="/mnt/mount-point-directory/datasets2")
	volume_mount_2 = client.V1VolumeMount(name='dshm', mount_path="/dev/shm")

	container=client.V1Container(name='gpu-train-container-'+experiment_name, 
		image='gcr.io/root-grammar-308418/sidewalk-train:v2',
		resources=requirement, volume_mounts=[volume_mount_1,volume_mount_2])
	#container.command=['sleep','3600']
	container.command=['sh','train.sh']
	container.args = [experiment_name]
	
	claim =client.V1PersistentVolumeClaimVolumeSource(claim_name='pvc-demo')
	volume_1=client.V1Volume(name='mypv',persistent_volume_claim=claim)

	empty_dir = client.V1EmptyDirVolumeSource(medium='Memory',size_limit='4Gi') #set medium='Memory' for RAM, "size_limit" argument in case memory is exceeded 
	volume_2=client.V1Volume(name='dshm', empty_dir=empty_dir)
	
	node_selector = {"cloud.google.com/gke-nodepool":"pool-gpu-train-a100"}

	spec = client.V1PodSpec(containers=[container], node_selector=node_selector, restart_policy="Never", volumes=[volume_1,volume_2])
	pod.spec = spec
	v1.create_namespaced_pod(namespace='default',body=pod)
	print(f"GPU Training Pod {experiment_name} deployed")
	return 'gpu-train-pod-'+experiment_name
	

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Arguments for gdal container pod')
	parser.add_argument('--city_name',type=str,required=False,default='welland',help='name of city to be processed')
	parser.add_argument('--tile_size',type=int,required=False,default=1250,help='tile size')
	parser.add_argument('--tile_stride',type=int,required=False,default=1100,help='tile stride')
	parser.add_argument('--model_name',type=str,required=False,default='experiment_9',help='name of model')
	parser.add_argument('--model_trained_multi_gpu',type=bool,required=False,default=True,help='multi gpu model?')
	parser.add_argument('--tile_resize_dim',type=int,required=False,default=768,help='tile resize dimention')
	parser.add_argument('--tile_start_idx',type=int,required=False,default=0,help='start index of tile')
	parser.add_argument('--tile_end_idx',type=int,required=False,default=99,help='end index of tile')
	parser.add_argument('--experiment_name',type=str,required=False,default='experiment-1000')

	args =parser.parse_args()

	load_kube_config()
	
	create_gdal_pod(city_name=args.city_name, tile_size=args.tile_size, tile_stride=args.tile_stride)
	
	#create_gpu_inference_pod(city_name=args.city_name, tile_size=args.tile_size, tile_stride=args.tile_stride,
	#	model_name=args.model_name,model_trained_multi_gpu=args.model_trained_multi_gpu,
	#	tile_resize_dim=args.tile_resize_dim, tile_start_idx=args.tile_start_idx, tile_end_idx=args.tile_end_idx,pod_number=0)

	# create_gpu_training_pod(args.experiment_name)

	#create_pre_training_pod(args.experiment_name)
