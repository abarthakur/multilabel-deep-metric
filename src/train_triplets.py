import mydatasets
import mymodels
import utils
import numpy as np
import torch
import sys
import os
import json 

args={
	"resume_from":0,
	"num_epochs":200,
	"checkpoint":20,
	"log":1,
	"hidden":[1000],
	"load_from":"",
	"nbrs":20,
	"emb_dim":20,
	"margin":0,
	"disc":0
	}

for i in range(1,len(sys.argv)):
	string=sys.argv[i]
	parts=string.split("=")
	assert(len(parts)==2)
	if parts[0]=="dataset":
		args["dataset_name"]=parts[1]
	elif parts[0]=="load":
		args["load_from"]=parts[1]
	elif parts[0]=="run_dir":
		dir_path=parts[1]
		if os.path.exists(dir_path):
			assert(os.path.isdir(dir_path))
		else:
			os.makedirs(dir_path)
		args["run_dir"]=parts[1]
	elif parts[0]in ["resume_from","num_epochs","emb_dim","checkpoint","log","nbrs","disc"]:
		args[parts[0]]=int(parts[1])
	elif parts[0]=="hidden":
		args["hidden"]=[int(k) for k in parts[1].split(",")]
	elif parts[0]=="val_file":
		args["val_file"]=parts[1]
	elif parts[0]=="margin":
		args["margin"]=float(parts[1])

# get data
x_mat,y_mat,x_tst,y_tst=mydatasets.get_dataset_for_exp(args["dataset_name"])
x_trn,y_trn,x_val,y_val=mydatasets.get_validation_split(x_mat,y_mat,args["val_file"],0.1)
num_labels=y_mat.shape[1]
num_dims=x_mat.shape[1]

factors=None
if args["disc"]==1:
	lcounts=np.sum(y_trn,axis=0)
	ranks=np.argsort(np.argsort(lcounts))+1
	factors=1/(np.log(ranks+1))
elif args["disc"]==2:
	lcounts=np.sum(y_trn,axis=0)
	factors=1/(np.log(lcounts+2))
else:
	factors=np.zeros(num_labels)+1.0
print(factors)
# model params
embedding_dim=args["emb_dim"]
model_layers=[num_dims]+args["hidden"]+[embedding_dim]
load_model_from=args["load_from"]

# training hyper params
resume_from=args["resume_from"]
num_epochs=args["num_epochs"]
checkpoint_every=args["checkpoint"]
log_every=args["log"]
batch_sizes=[(0,500)]
max_triplets_per_anchor=[(0,100),(50,75),(200,200)]
max_negatives_per_positive=[(0,3),(10,6),(30,12)]
trips_margin=[(0,args["margin"])]
lrates=[(0,0.01),(100,0.001)]
# testing hyper params
num_neighbours=args["nbrs"]
#log file

logfilename=args["run_dir"]+"/log.txt"

def log_experiment_parameters():
	with open(logfilename,"a") as fi:
		fi.write(json.dumps({
		"model_layers":model_layers,
		"embedding_dim":embedding_dim,
		"load_model_from":load_model_from,
		"resume_from":resume_from,
		"num_epochs":num_epochs,
		"checkpoint_every":checkpoint_every,
		"batch_sizes":batch_sizes,
		"max_triplets_per_anchor":max_triplets_per_anchor,
		"max_negatives_per_positive":max_negatives_per_positive,
		"trips_margin":trips_margin,
		"lrates":lrates,
		"num_neighbours":num_neighbours,
		"disc":args["disc"]
		}))
		fi.write("\n######\n")

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
		trips=utils.get_triplets(embeddings,y_batch,max_neg,max_trips,factors,debug=False)
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
	if (epoch+1)%log_every==0:
		utils.log_epoch_metrics(logfilename,epoch,loss_mean,model,x_trn,y_trn,x_val,y_val,num_neighbours)
	# evaluate model
	if (epoch+1)%checkpoint_every==0:
		torch.save(model,args["run_dir"]+"/model_"+str(epoch+1))
