from cvxpy import * #http://www.cvxpy.org/en/latest/
import numpy as np
from numpy.matlib import *
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_data(sz):
	sz = int(sz / 2)
	Ax = np.random.randint(20, high=31, size=sz) / 100
	Bx = np.random.randint(50, high=61, size=sz) / 100	
	Ay = np.random.rand(sz)
	By = np.random.rand(sz)
	features = np.vstack((np.column_stack((Ax,Ay)),np.column_stack((Bx,By))))
	tags = np.empty(features.shape[0], dtype=object)
	tags[features[:,1] > 0.5] = 'D'
	tags[features[:,1] <= 0.5] = 'C'
	tags[features[:,0] < 0.4] = 'A'
	return features, tags

def get_root(g):
	return nx.topological_sort(g)[0]

def Optimization(data, hierarchy, fraction_novel=None, root=None):
	# If root is none in the DAG you can spare some time.
	# Create DAG from hierarchy, i.e. a list of tuples representing edges
	g = nx.DiGraph()
	g.add_edges_from(hierarchy)
	features = data['features']
	labels = data['tags']	
	n = features.shape[0] #number of samples
	d = features.shape[1] #number of features   
	k = g.number_of_nodes() #depth of hierarchy
	#Note that the graph must be without cycles to be a DAG.
	if not root:
		root = get_root(g)	
	t_order = list(nx.bfs_tree(g, root))
	numerical_hierarchy = []
	# Convert hierarchy to numeric indexes
	for i , (parent, child) in enumerate(hierarchy): 
		numerical_hierarchy.append((t_order.index(parent), t_order.index(child)))
	# Convert labels to numerical labels
	numerical_labels = np.empty(labels.shape)
	for i , label in enumerate(labels):
		numerical_labels[i] = t_order.index(label)
	# Variables
	R = Variable(k)
	means = Variable(d,k)	
	# Constraints
	constraints = [0 <= R]
	# Append other constraints.
	for i in range(k-1):
		constraints.append(norm(means[:,numerical_hierarchy[i][0]]-means[:,numerical_hierarchy[i][1]])\
				<= R[numerical_hierarchy[i][0]] - R[numerical_hierarchy[i][1]])
	if fraction_novel:
		xi = Variable(n)
		# Parameters
		n_novel = n * fraction_novel;
		C = k / n_novel;
		# Constraints for slacks
		constraints.append(0 <= xi)
		for i in range(n):
			constraints.append(norm(features[i,:]-means[:,numerical_labels[i]]) \
				<= R[numerical_labels[i]] + xi[i])
			# Problem
		prob = Problem(Minimize(sum_entries(R) + C*sum_entries(xi)), constraints)
	else:
		for i in range(n):
			constraints.append(norm(features[i,:]-means[:,numerical_labels[i]]) \
				<= R[numerical_labels[i]])
			#Problem
		prob = Problem(Minimize(sum_entries(R)), constraints)	
	
	#Solve, but consider computation times when number of samples is relevant
	print('Constraints set, starting optimization..')
	try:
		import mosek
		prob.solve(solver=MOSEK, verbose=True)
		# prob.solve(solver=MOSEK, verbose=True, mosek_params={'MSK_IPAR_LOG_FEAS_REPAIR': 1e-20})
		# mosek_params eat a dictionary as argument. 
		# C++ API details are here: http://docs.mosek.com/7.1/toolbox/Parameters.html	
	except ImportError:		
		prob.solve(solver=SCS, verbose=True)
	output = {}
	output['Radii'] = R
	output['Means'] = means 
	if fraction_novel:
		output['Slacks'] = xi
	return output

if __name__ == "__main__":
	data = {}
	hierarchy = [('root', 'A'), ('root', 'B'), ('B', 'C'), ('B', 'D')]
	g = nx.DiGraph()
	g.add_edges_from(hierarchy)
	t_order = list(nx.bfs_tree(g,'root'))
	features, tags = generate_data(200)
	swell_features = []
	swell_labels = []
	for i, l in enumerate(tags):
		temp = nx.all_simple_paths(g, 'root', l)
		for t in temp: 
			swell_labels = swell_labels + t
			swell_features.append(repmat(features[i],len(t),1))
	swell_features = np.concatenate(swell_features)
	swell_labels = np.array(swell_labels, dtype='object')
	data['features'] = swell_features
	data['tags'] = swell_labels
	out = Optimization(data, hierarchy)
	# Draw DAG
	nx.draw_networkx(g, with_labels=True)
	# Plot problem
	fig, ax = plt.subplots(1)
	available_tags = np.unique(tags)
	colors = 'bgrcmykw'
	patchies = []
	for i, k in enumerate(t_order):
		if k in available_tags:
			ax.scatter(features[tags==k][:,0], features[tags==k][:,1],\
			c=colors[i])	
		patchies.append(patches.Patch(color=colors[i], label=k))	
	ax.legend(handles=patchies, ncol=len(patchies), loc='lower left',\
				bbox_to_anchor=(0.1, -0.4))
	for i, l in enumerate(t_order):
		centroid = out['Means'][:,i].value
		radius = float(out['Radii'][i].value)	
		circ = patches.Circle(centroid, radius, color=colors[i], fill=False)
		ax.add_patch(circ)	
	ax.set_xlim(-0.2, 1.1)
	ax.set_ylim(-0.1, 1.1)
	ax.set_aspect(1)
	ax.legend(handles=patchies, ncol=len(patchies), loc='upper center',\
				bbox_to_anchor=(0.5, -0.05))	
	plt.show()
