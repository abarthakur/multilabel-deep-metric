import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn import metrics as skmet
import torch

def make_batches(x_mat,y_mat,batch_size,shuffle=True):
	if shuffle:
		shuffled_order=np.random.permutation(x_mat.shape[0])
		x_mat=x_mat[shuffled_order,:]
		y_mat=y_mat[shuffled_order,:]
	train_dataset=[]
	start=0
	while start + batch_size <= x_mat.shape[0]:
		end=min(start+batch_size,x_mat.shape[0])
		train_dataset.append((x_mat[start:end,:],
							y_mat[start:end,:]))
		start+=batch_size
	return train_dataset


def select_triplets(embeddings,y_batch,threshold_similarity,num_positives,max_negatives,debug=False):
	triplets=[]
	for i in range(0,embeddings.shape[0]):
		anchor=embeddings[i,:]
		anchor_y=y_batch[i,:]
		sim_scores=np.sum(y_batch * anchor_y,axis=1)
		thres=np.minimum(sim_scores[i]-1,threshold_similarity)
		thres=np.maximum(thres,0)
		dists=np.sqrt(np.sum((embeddings-anchor)**2,axis=1))
		distance_order=np.argsort(dists)
		batch_order_from_distance_order=np.argsort(distance_order)
		sim_dist_wise=sim_scores[distance_order]
		if debug:
			print(i,num_positives)
			print(sim_scores)
		# idcs are in distance order
		negative_idcs=np.nonzero(sim_dist_wise<=thres)[0]
		# mine negatives first
		for j,neg_idx in enumerate(negative_idcs):
			if j > max_negatives:
				break
			positive_idcs=np.nonzero(sim_dist_wise[neg_idx+1:]>thres)[0]
			# end of the queue
			if len(positive_idcs)==0:
				break
			positive_idcs+=neg_idx+1
			# get positives
			for k in range(0,np.minimum(num_positives,positive_idcs.shape[0])):
				t=np.random.randint(0,positive_idcs.shape[0])
				pos_idx=positive_idcs[t]
				triplets.append((i,
								distance_order[pos_idx],
								distance_order[neg_idx]))
				if debug:
					print((i,distance_order[pos_idx],distance_order[neg_idx]))				
	return triplets

def get_triplets(embeddings,y_batch,threshold_similarity,num_positives,max_negatives,debug=False):
	trips_list=select_triplets(embeddings.detach().numpy(),y_batch,threshold_similarity,
								num_positives,max_negatives,debug)
	if len(trips_list)==0:
		return None
	anchors=[]
	positives=[]
	negatives=[]
	for (a,p,n) in trips_list:
		anchors.append(embeddings[a,:].reshape(1,embeddings.shape[1]))
		positives.append(embeddings[p,:].reshape(1,embeddings.shape[1]))
		negatives.append(embeddings[n,:].reshape(1,embeddings.shape[1]))
	anchors=torch.cat(anchors,0)
	positives=torch.cat(positives,0)
	negatives=torch.cat(negatives,0)
	return anchors,positives,negatives


def scheduled_value(epoch,list_of_vals):
	sched_val=None
	for (st,val) in list_of_vals:
		if st<=epoch:
			sched_val=val
	return sched_val


def precision_at_k(y_tst,probs_pred,k):
	assert(k>0 and int(k)==k)
	assert(y_tst.shape[0]==probs_pred.shape[0])
	top_k=np.argsort(probs_pred,axis=1)[:,-k:]
	total=0
	for s_idx in range(0,y_tst.shape[0]):
		best_labels=top_k[s_idx,:]
		total+=np.sum(y_tst[s_idx,best_labels])
	p_at_k=total/y_tst.shape[0]
	p_at_k=p_at_k/k
	return p_at_k


def save_metrics(emb_trn,y_trn,emb_val,y_val,num_neighbours,filename,log=True):
	nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(emb_trn)
	nbr_distances, nbr_indices = nbrs.kneighbors(emb_val)
	y_nbr=y_trn[nbr_indices,:]
	assert(y_nbr.shape==(emb_val.shape[0],num_neighbours,y_val.shape[1]))
	y_pred_val=np.mean(y_nbr,axis=1)
	metrics_df=pd.DataFrame()
	metrics_df.loc[0,"p@1"]=precision_at_k(y_val,y_pred_val,1)
	metrics_df.loc[0,"p@3"]=precision_at_k(y_val,y_pred_val,3)
	metrics_df.loc[0,"p@5"]=precision_at_k(y_val,y_pred_val,5)
	metrics_df.loc[0,"ranking_loss"]=skmet.label_ranking_loss(y_val,y_pred_val)
	metrics_df.loc[0,"coverage_error"]=skmet.coverage_error(y_val,y_pred_val)
	metrics_df.loc[0,"avg_prec_score"]=skmet.label_ranking_average_precision_score(y_val,y_pred_val)
	if log:
		print(metrics_df)
	if filename:
		with open(filename,"a") as fi:
			metrics_df.to_csv(fi)


