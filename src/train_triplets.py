import mydatasets
import mymodels
import utils
import numpy as np
import torch

# data params
DATASET="bibtex"
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
run_name="results/bibtex_test"
embedding_dim=30
model_layers=[num_dims,1000,embedding_dim]
load_model_from=""

# training hyper params
resume_from=0
num_epochs=200
checkpoint_every=50
log_every=1
batch_sizes=[(0,500)]
max_triplets_per_anchor=[(0,100),(50,75),(200,200)]
max_negatives_per_positive=[(0,3),(10,6),(30,12)]
trips_margin=[(0,0)]
lrates=[(0,0.01),(100,0.001)]
# testing hyper params
num_neighbours=20
#log file

logfilename=run_name+"_log.txt"

def log_experiment_parameters():
	with open(logfilename,"a") as fi:
		fi.write(
		"model_layers : " + str(model_layers) +"\n"+
		"embedding_dim : " + str(embedding_dim) +"\n"+
		"load_model_from : " + str(load_model_from) +"\n"+
		"resume_from : " + str(resume_from) +"\n"+
		"num_epochs : " + str(num_epochs) +"\n"+
		"checkpoint_every : " + str(checkpoint_every) +"\n"+
		"batch_sizes : " + str(batch_sizes) +"\n"+
		"max_triplets_per_anchor : " + str(max_triplets_per_anchor) +"\n"+
		"max_negatives_per_positive : " + str(max_negatives_per_positive) +"\n"+
		"trips_margin : " + str(trips_margin) +"\n"+
		"lrates : " + str(lrates) +"\n"+
		"num_neighbours : " + str(num_neighbours) +"\n"
		)

# get model
if load_model_from:
	model=torch.load(load_model_from)
else:
	model=mymodels.SimpleNet(num_dims,embedding_dim,model_layers)

log_experiment_parameters()

# training loop
for epoch in range(resume_from,num_epochs):
	# get scheduled values of hyper params
	tmargin=utils.scheduled_value(epoch,trips_margin)
	batch_size=utils.scheduled_value(epoch,batch_sizes)
	lrate=utils.scheduled_value(epoch,lrates)
	max_trips=utils.scheduled_value(epoch,max_triplets_per_anchor)
	max_neg=utils.scheduled_value(epoch,max_negatives_per_positive)
	print("Epoch ",epoch,(tmargin,batch_size,lrate,max_trips,max_neg))
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
		trips=utils.get_triplets(embeddings,y_batch,max_neg,max_trips,debug=False)
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
	if epoch%log_every==0:
		utils.log_epoch_metrics(logfilename,epoch,loss_mean,model,x_trn,y_trn,x_val,y_val,num_neighbours)
	# evaluate model
	if epoch%checkpoint_every==0:
		torch.save(model,run_name+"_"+str(epoch))
