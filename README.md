# Amazon Computers GNN

#### -- Project Status: Completed

## Project Intro/Objective
Graph Neural Networks (GNNs) are a family of Neural Networks that operate on a Graph structure and make complex graph data easy to understand. In this project, I will be implementing a GNN model to classify nodes in the Amazon Computers dataset. In this dataset, nodes represent goods, edges indicate that two goods are frequently bought together, node features are bag-of-words encoded product reviews, and class labels are given by the product category.

### Methods Used
* Deep Learning
* Graph Neural Networks (GNNs)
* Dense Neural Networks (DNNs)


### Technologies
* Python
* Pandas, jupyter
* scikit-learn
* PyTorch
* PyTorch Geometric


## Project Description
The Amazon Computers dataset was first introduced in [Shchur et al., 2019](https://arxiv.org/abs/1811.05868) titled <i>Pitfalls of Graph Neural Network Evaluation</i>. The dataset is a segment of the Amazon co-purchase graph [McAuley et al., 2015](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf). The dataset consists of 13,752 nodes and 491,722 edges. The graph is undirected, isolated nodes do exist, and the average node degree is 35.8.

First, I try a simple MLP model with 3 Linear Layers, which uses only feature embeddings to predict class labels. The model continues to train until around 150 epochs where it begins overfitting. At this point, the ROC AUC is roughly 0.55. Then I try a GNN model with 3 GCNConv Layers. The model continues to train until around 1,500 epochs before it tarts overfitting and achieves a ROC AUC of around 0.76. The difference in model performance shows that the model was able to take advantage of the available information in the edge features, which allowed it to train for longer without overfitting and achieve much better performance.


## Featured Notebooks/Analysis/Deliverables
* [Notebook](https://github.com/lukemonington/AmazonComputersGNN/blob/master/main_ai.ipynb)


## Contributing Members

**[Luke Monington](https://github.com/lukemonington)**

## Contact
* I can be reached at lukemonington3@gmail.com.
