import mydatasets
import mymodels
import utils
import numpy as np
import torch

# data params
DATASET="mediamill"
NUM_MONITOR=10

# get data
x_mat,y_mat,x_tst,y_tst=mydatasets.get_dataset_for_exp(DATASET)
x_trn,y_trn,x_val,y_val=mydatasets.get_validation_split(x_mat,y_mat,"./"+DATASET+"datadict.p",0.1)
num_labels=y_mat.shape[1]
num_dims=x_mat.shape[1]
spc_idcs=np.random.permutation(x_trn.shape[0])[:NUM_MONITOR]
x_special=x_trn[spc_idcs,:]
y_special=y_trn[spc_idcs,:]

# model params
run_name="results/mediamill_run_1"
embedding_dim=50
model_layers=[num_dims,1000,embedding_dim]
load_model_from=""

# training hyper params
resume_from=0
num_epochs=300
checkpoint_every=5
batch_sizes=[(0,500)]
similarity_thresholds=[(0,0),(200,1),(220,2),(250,4)]
num_positives=[(0,3),(100,5),(200,8)]
max_negatives=[(0,10),(100,20),(200,30)]
trips_margin=[(0,0.003)]
lrates=[(0,0.01),(100,0.001)]

# testing hyper params
num_neighbours=10

# get model
if load_model_from:
	model=torch.load(load_model_from)
else:
	model=mymodels.SimpleNet(num_dims,embedding_dim,model_layers)

# training loop
for epoch in range(resume_from,num_epochs):
	# get scheduled values of hyper params
	tmargin=utils.scheduled_value(epoch,trips_margin)
	batch_size=utils.scheduled_value(epoch,batch_sizes)
	lrate=utils.scheduled_value(epoch,lrates)
	thres=utils.scheduled_value(epoch,similarity_thresholds)
	num_pos=utils.scheduled_value(epoch,num_positives)
	max_neg=utils.scheduled_value(epoch,max_negatives)
	print("Epoch ",epoch,(tmargin,batch_size,lrate,thres,num_pos,max_neg))
	# define loss and create optimizer
	triplet_loss = torch.nn.TripletMarginLoss(margin=tmargin, p=2)
	optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
	# get batches
	mini_batches=utils.make_batches(x_trn,y_trn,batch_size)
	loss_values=[]
	for batch_num,batch in enumerate(mini_batches):
		x_batch,y_batch=batch
		# generate embeddings
		embeddings=model(torch.from_numpy(x_batch.astype('float32')))
		# generate triplets (online)
		trips=utils.get_triplets(embeddings,y_batch,thres,num_pos,max_neg)
		if trips is None:
			continue
		anch,pos,neg=trips
		# compute loss
		loss_batch=triplet_loss(anch,pos,neg)
		loss_values.append(loss_batch.detach().numpy())
		# backprop
		optimizer.zero_grad()
		loss_batch.backward(retain_graph=True)
		optimizer.step()
		print("Batch size :",anch.shape[0],", Loss value :",loss_batch.detach().numpy())
	loss_mean=np.mean(np.array(loss_values))
	print("Loss for this epoch :",loss_mean)
	# evaluate model
	if epoch%checkpoint_every==0:
		emb_trn=model(torch.from_numpy(x_trn.astype('float32'))).detach().numpy()
		emb_val=model(torch.from_numpy(x_val.astype('float32'))).detach().numpy()
		utils.save_metrics(emb_trn,y_trn,emb_val,y_val,num_neighbours,
							run_name+"_metrics.csv",log=True)
		torch.save(model,run_name+"_"+str(epoch))
