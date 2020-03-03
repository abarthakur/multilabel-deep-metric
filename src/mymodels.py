import torch
import torch.nn.functional as F

class SimpleNet(torch.nn.Module):
	def __init__(self,input_dim,output_dim,linear_widths):
		super(SimpleNet, self).__init__()
		self.linear_list=torch.nn.ModuleList([])
		assert(input_dim==linear_widths[0])
		assert(output_dim==linear_widths[-1])
		#simple network of successive linear layers
		for i in range(1,len(linear_widths)):
			inp_dim=linear_widths[i-1]
			out_dim=linear_widths[i]
			new_layer=torch.nn.Linear(inp_dim,out_dim)
			#initialize weights and bias    
			torch.nn.init.xavier_normal_(new_layer.weight, gain=1)
			new_layer.bias.data.fill_(0.01)
			self.linear_list.append(new_layer)


	#use relu or sigmoid?
	def forward(self,input_sample):
		x=input_sample
		for layer in self.linear_list:
			x = F.relu(layer(x))
		#normalize to restrict to unit hypersphere as in paper
		x = F.normalize(x, p=2, dim=1)#??
		return x