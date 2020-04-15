
import numpy as np
import pandas as pd
import pickle
import os

## Paths to files

def load_dataset(dataset_name,debug=True):
	if dataset_name in ["mediamill","delicious","bibtex"]:
		return load_small_dataset(dataset_name,debug)
	else:
		return load_large_dataset(dataset_name,debug)


def load_large_dataset(dataset_name,debug=True):
	smalls=dataset_name.lower()
	train_path="../data/"+smalls+"/"+smalls+"_train.txt"
	test_path="../data/"+smalls+"/"+smalls+"_test.txt"
	trn_data=read_data_file(train_path,debug)
	tst_data=read_data_file(test_path,debug)
	return trn_data,tst_data


def load_small_dataset(dataset_name,debug=True):

	# check input
	dataset_names=["mediamill","delicious","bibtex"]
	if dataset_name not in dataset_names:
		print("Invalid input")
		return
	firstcap=dataset_name.capitalize()

	# path names
	data_path="../data/"  +  firstcap +"/"+ firstcap+"_data.txt"
	trsplit_path="../data/"  +  firstcap +"/"+ dataset_name+"_trSplit.txt"
	tstsplit_path="../data/"  +  firstcap +"/"+ dataset_name+"_tstSplit.txt"

	if debug:
		print("Loading datasets")
		print(data_path)
		print(trsplit_path)
		print(tstsplit_path)
	trsplits,tst_splits=read_split_files(trsplit_path,tstsplit_path,debug)
	return read_data_file(data_path,debug),trsplits,tst_splits

def read_data_file(data_path,debug):
	# Read header and lines (points)
	header=None
	lines=[]
	with open(data_path,"r") as daf:
		header=daf.readline()
		for l in daf:
			lines.append(l)
	num_points, num_features, num_labels =[int(x) for x in header.split()]
	if debug:
		print("## HEADER ##")
		print("#Point :",num_points,", #Features :",num_features,", #Labels :",num_labels)
	assert num_points==len(lines), "header num_points doesn't match with num_lines of file"

	# Parse lines to extract labels & features which are specified as 
	# [ { feats = [ (f1,v1),... ] } , labels = [ l1,...] },... ]

	import re
	all_points=[]
	for i,line in enumerate(lines):
		point={}
		
		#line begins with comma separated labels
		match=re.search(r"\A((\d*,)*(\d+))\s",line)
		labstring=""
		labels=[]
		if match :
			labstring=match.groups()[0]
			labels=[int(l) for l in labstring.split(",")]
		
		#followed by f1:v1 f2:v2 ...
		featstring=line.replace(labstring,"",1).strip()
		feats= featstring.split()
		feats =[f.split(":") for f in feats]
		feats= [(int(f[0]),f[1]) for f in feats]
		
		#check values are as expected
		#labels
		assert(len(labels)<=num_labels)
		if len(labels)>0:
			assert(max(labels)<num_labels)
			assert(min(labels)>=0)   
		#feats   
		feat_idcs= [f[0] for f in feats]
		assert(len(feats)<=num_features)
		assert(max(feat_idcs)<num_features)
		assert(min(feat_idcs)>=0)
		
		#fill list
		point["labels"]=labels    
		point["features"]=feats
		all_points.append(point)

	#cautionary
	assert(len(all_points)==num_points)

	# Read features into numpy array
	x_mat=np.zeros((num_points,num_features),dtype=float)
# 	x_mat[:]=np.nan
	for i,p in enumerate(all_points):
		for f_idx,f_val in p["features"]:
			x_mat[i][f_idx]=f_val

	# Read labels into numpy array
	y_mat=np.zeros((num_points,num_labels),dtype=int)
	for i,p in enumerate(all_points):
		for l in p["labels"]:
			y_mat[i][l]=1

	# Create dataframe
	full_dataset =pd.DataFrame({"features":[x_mat[i,:] for i in range(0,num_points)]
		,"labels_binary" : [y_mat[i,:] for i in range(0,num_points)]
		, "labels_list" : [all_points[i]["labels"] for i in range (0,num_points)]
		})

	return full_dataset

def read_split_files(trsplit_path,tstsplit_path,debug):
	# Read training split files into pandas dataframes
	trsplits=pd.read_csv(trsplit_path,header=None,delim_whitespace=True)
	tstsplits=pd.read_csv(tstsplit_path,header=None,delim_whitespace=True)
	#original files are 1 indexed 
	trsplits=trsplits-1
	tstplits=tstsplits-1
	num_splits=len(trsplits.columns)
	assert(len(tstsplits.columns)==num_splits)
	if debug:
		print("Number of splits :",num_splits)
	return trsplits,tstplits

def get_mulan_arrays(dataset_name):
	with open("../data/"+dataset_name+"/data_dict.p","rb") as fi:
		data_dict=pickle.load(fi)
		return data_dict["x_trn"],data_dict["y_trn"],data_dict["x_tst"],data_dict["y_tst"]


def get_small_dataset_split(dataset,trn_splits,tst_splits,split_num):
	assert(split_num<len(trn_splits))
	trn_data=dataset.iloc[trn_splits[split_num].to_numpy()]
	tst_data=dataset.iloc[tst_splits[split_num].to_numpy()]
	trn_data=trn_data.reset_index(drop=True)
	tst_data=tst_data.reset_index(drop=True)
	return trn_data,tst_data


def get_arrays(trn_data,tst_data):
	y_mat=np.vstack(trn_data["labels_binary"].to_numpy())
	x_mat=np.vstack(trn_data["features"].to_numpy())
	y_tst=np.vstack(tst_data["labels_binary"].to_numpy())
	x_tst=np.vstack(tst_data["features"].to_numpy())
	return x_mat,y_mat,x_tst,y_tst


def get_dataset_for_exp(DATASET,SPLIT=0,remove_unlabelled=True):
	# change dirs because paths are hardcoded in mydatasets
	if DATASET in ["delicious","mediamill","bibtex"]:
		full_dataset,trn_splits,tst_splits=load_small_dataset(DATASET)
		trn_data,tst_data=get_small_dataset_split(full_dataset,trn_splits,tst_splits,SPLIT)
		x_mat,y_mat,x_tst,y_tst=get_arrays(trn_data,tst_data)
	elif DATASET in ["eurlex"]:
		trn_data,tst_data=load_large_dataset(DATASET)
		x_mat,y_mat,x_tst,y_tst=get_arrays(trn_data,tst_data)
	elif DATASET in ["enron","yeast"]:
		x_mat,y_mat,x_tst,y_tst=get_mulan_arrays(DATASET)
	# remove nz samples
	nz_samples=np.sum(y_mat,axis=1)!=0
	x_mat_new=x_mat[nz_samples,:]
	y_mat_new=y_mat[nz_samples,:]
	print(x_mat_new.shape)
	nz_samples=np.sum(y_tst,axis=1)!=0
	x_tst_new=x_tst[nz_samples,:]
	y_tst_new=y_tst[nz_samples,:]
	print(x_tst_new.shape)
	return x_mat_new,y_mat_new,x_tst_new,y_tst_new

def get_validation_split(x_mat,y_mat,filename,valsplit=0.1):
	if os.path.exists(filename):
		with open(filename,"rb") as fi:
			data_dict=pickle.load(fi)
		trn_idcs=data_dict["trn"]
		val_idcs=data_dict["val"]
	else:
		num_val=int(x_mat.shape[0]*valsplit)
		perm=np.random.permutation(x_mat.shape[0])
		trn_idcs=perm[num_val:]
		val_idcs=perm[:num_val]
		with open(filename,"wb") as fi:
			data_dict={"trn":trn_idcs,"val":val_idcs}
			pickle.dump(data_dict,fi)	
	x_val=x_mat[val_idcs,:]
	y_val=y_mat[val_idcs,:]
	# replace
	x_trn=x_mat[trn_idcs,:]
	y_trn=y_mat[trn_idcs,:]
	return x_trn,y_trn,x_val,y_val