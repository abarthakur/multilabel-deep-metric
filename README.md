### Project Overview

This project adapts triplet loss based metric learning to learn a metric for multilabel points, such that samples with maximum overlap in label sets are close. Challenges were appropriately defining the mining of triplets, as there is significant overlap between classes in multilabel classification.

![Image of Embeddings for Bibtex dataset generated after 40 epochs of training](/cover_bibtex_emb_30_marg_0_h_1000_ep_40.png)

### Project Description

We want to learn a transformation that converts features to an embedding space, such that points with overlapping label sets are placed close to each other. That is, we want to learn a transformation where distance between two points is inversely proportional to the number of labels they share. 

**Task** : Multi Label Ranking/ Multi Label Classification with large number of labels.

**Objective** : Given features, learn an embedding space, where a points neighbours (in terms of Euclidean distance) have maximum overlap in labelsets.

#### Network and Loss

* We use a dense network with a single hidden layer as the metric function. Preliminary experiments do not show any improvements in increasing number of hidden layers. Moreover, in a directly supervised learning setting (where we try to predict labels as targets), single hidden layer networks performed the best by our experiments. In a supervised learning setting, this can be attributed to the sparsity and high dimensionality of the features - various works have shown that SVMs and other linear classifiers perform better in such domains - which allows greater linear separability. 

* We use the Triplet Loss to train our network. Other possible choices are Contrastive Loss and Magnet Loss, but we choose triplet loss mainly because it has had quite a bit of empirical success in the related domain of multi-class classification with large number of classes (for example person reidentification) as well as ranking images. 

#### Triplet Mining Method

In the multi label setting, there isn't such a clear definition of positive and negative, but we just need the "similarities". A nice proxy is the number of labels two objects have in common, which can be simply calculated as ``y_1 . y_2``. 

Thus let us define,

```
d(z_1,z_2)=||z_a-z_i||^2 (predicted metric)
d(y_1,y_2)=-y_1.y_2. (target metric)
```

We define a **valid triplet** as an ordered tuple (a,p,n) such that ``d(z_a,z_p) + margin > d(z_a,z_n) AND d(y_a,y_p) < d(y_a,y_n)``. 

Consider the following example for illustration, for a margin=0

* Consider a point A as the anchor, and 4 other points B,C,D,E in the minibatch. Let's assume they are ordered such that distance to A increases from B to E (in embedding space). 
* Suppose their similarities/labelset overlaps be however [4,1,2,5]. Ideally, we want it to look like [5,4,2,1]. Thus we want to reorder B-E w.r.t. distance from A to E,B,D,C.
* In this example our triplets are (A,D,C),(A,E,B),(A,E,C),(A,E,D).

We develop our triplet mining method in the following way

1. We use an online strategy, mining triplets from a minibatch similar to []. This is an obvious choice due to memory constraints.
2. We divide our training data into equally sized minibatches randomly. We could try to construct minibatches so that pairs which are highly relevant fall in the same minibatch, but this task will be increasingly complex as the number of labels increase. So we stick to the simpler method, although this potentially results in a serious flaw : the chance of a rare label occuring multiple times in a minibatch is unlikely, thus this method potentially ignores intra-class similarities for rare labels. But we roll with it.
3. Since number of triplets vary cubically with batch size, selecting all valid triplets is generally infeasible.
4. In the first step, we choose all points within a minibatch as anchors, and we choose all points with non-zero similarities as positives. For each anchor-positive pair, we choose ``max_negatives_per_pos`` number of negatives(0-similarity) which are closer than the positive. This is somewhat analogous to "category-level" triplets as described [Wang 2014][1].
5. In the second step, we choose all misorderings of all samples with positive similarity as triplets. Since we expect the number of points within a minibatch with overlap with the anchor to be low for multilabel datasets, this step is not infeasible. This step is (again, somewhat) analogous to "fine grained" triplets as described in [Wang 2014][1].

```
for each point as anchor
	for each pos such that sim(anchor,pos)>0:
		1. choose all points sim(anchor,neg)>0, and d(a,n)<d(a,p)
		2. choose uniformly at random K points with sim(anchor,neg)==0, and d(a,n)<d(a,p)
```

#### Files and Installation

This project was developed in Python 3.6.9 with the following libraries with versions

```
numpy                             1.17.4
scikit-learn                      0.22
torch                             1.4.0
torchvision                       0.5.0
# optional (for notebooks)
matplotlib                        3.1.2
seaborn                           0.9.0
```


``src/train_triplets.py`` contains the main training loop to train the network

```
cd src
python train_triplets.py run_dir=runs/bibtex_30_3500 dataset=bibtex val_file=runs/bibtex_datadict.p emb_dim=30 disc=0 margin=0 num_epochs=160 hidden=3500 checkpoint=20 log=1 nbrs=20
```

Only dataset, run_dir and val_file are required args.

Notebooks 

* [src/notebooks/training_plots.ipynb](/src/notebooks/training_plots.ipynb) : Comparision of various runs with different hyperparameter values.
* [src/notebooks/embed_viz.ipynb](/src/notebooks/embed_viz.ipynb)`` : TSNE visualization of the embeddings.
* [src/notebooks/evaluation_model.ipynb](/src/notebooks/evaluation_model.ipynb)`` : Compares a simple averaging nearest-neighbour model vs a distance weighted one.

Utility files 

* ``src/utils.py`` : contains various helper functions, most importantly, the triplet mining method.
* ``src/mymodels.py`` : contains class definition for a simple feedforward NN.
* ``src/mydatasets.py`` : contains various helper functions to read and parse datasets downloaded from XCV repository into numpy arrays.

### Glossary

#### Binary, Multiclass and Multilabel Classification

**Binary classification** deals with estimating P(Y|X), where Y belongs to {0,1}

**Multiclass classification** deals with estimating P(Y|X) where Y belongs to {0,1,...,k}

Finally, **multilabel classification** deals with estimating P(Y|X) where Y belongs to Powerset({0,1,...,k}). Essentially, a binary partition of the labels into "relevant" and "not relevant"

**Multilabel ranking** estimates a ranking/ordering of the labels. This can be thresholded, to produce a binary partition of labels. Thus multilabel ranking can be thought of as a superset problem of multilabel classification. In practice, a point in a multi label datasets is annotated with a set of relevant labels, and not with a ranking over the labels. Thus ideally, when we are approaching the problem of MLC from a ranking perspective, we want the scores defining the ranking to be split cleanly into two groups with a margin. 

#### Deep Metric Learning

* Given two objects, we can always ask what is the distance between them? Or equivalently, how similar are they? We ask these questions in a machine learning setting, typically because we want to retrieve "similar"/close objects (in our training set) for a given novel object (a test case). 
* An obvious way to compute the distance between them is to take the Euclidean or any other standard distance function between their respective features/representations. Thus, given a novel object's representation, we simply find the closest objects in our training set in terms of euclidean distance in the representation space.
* But sometimes this isn't very useful, as this assumes that the objects are already arranged in this representation space according to the measure of similarity we care about.
* However, suppose we have, typically from another source, the "true" (for our purposes) distances between objects for our training set.
* Assuming that this is a complicated function of the representations that we do have, we can however try to learn this distance function. 
* This is essentially metric learning. We could directly learn ``f(x_1,x_2)=c`` directly. Alternatively, we can learn it implicitly by transforming ``x -> z`` , a space where the ``d(f(x_1),f(x_2))=c``. 

> Deep metric learning is when we use a neural network to approximate f. Most methods take the second approach of learning the metric implicitly by transforming the features to an "embedding" space.

#### Triplet Loss

A simple formulation for deep metric learning is to pick any two points, and learn a regression model ``f(x_1,x_2) -> d``. Alternatively, we can learn the distance implicitly ( ``d(f(x_1),f(x_2) -> d`` ), but still considering only 2 points. 

Triplet loss instead considers 3 points, the first is the anchor, the second positive (relative to the anchor) and the third negative (again, relative to the anchor). Then it minimizes the following loss

```
L = max{(d(f(x_a),f(x_p)) + margin) - d(f(x_a),f(x_n)),0}
```

Thus, while in the first case (also called Contrastive Loss), we choose the value of d to be 0 for like pairs, and some large number for unlike pairs, in triplet loss, we loosen the constraint somewhat and require only that like pairs are closer (by a margin) then unlike pairs. 

Uses of triplet loss for classification/ranking can be found in [Wang 2014][1] and [Schroff 2015][2]. A crucial step in triplet loss based methods, is mining triplets. We choose an online triplet mining scheme adapted from [Facenet][2].

* In [Facenet][2], triplets are chosen from a minibatch in the following way

```
for each point as anchor
	for each positive of anchor
		choose all negatives which are within d(a,p)+margin (hard negatives)
```

* Moreover, in the start of the training, they choose negatives which lie in (d(a,p),d(a,p)+margin). These are "semi-hard" negatives.

### References

* [The Extreme Classification Repository: Multi-label Datasets & Code](http://manikvarma.org/downloads/XC/XMLRepository.html)
* [Learning Fine-grained Image Similarity with Deep Ranking][1]
* [FaceNet: A Unified Embedding for Face Recognition and Clustering][2]

[1]: https://arxiv.org/abs/1404.4661 "Learning Fine-grained Image Similarity with Deep Ranking"
[2]: https://arxiv.org/abs/1503.03832 "FaceNet: A Unified Embedding for Face Recognition and Clustering"
