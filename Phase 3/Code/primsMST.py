import sys

class Graph(): 

	def __init__(self, vertices): 
		self.V = vertices 
		self.graph = [[0 for column in range(vertices)] 
					for row in range(vertices)] 

	# A utility function to print the constructed MST stored in parent[] 
	def createMST(self, parent): 
		result = []
		for i in range(1,self.V):
			result.append((parent[i], i, self.graph[i][parent[i]]))
		return result
            
	def minKey(self, key, mstSet): 

		# Initilaize min value 
		min = float("inf")

		for v in range(self.V): 
			if key[v] < min and mstSet[v] == False: 
				min = key[v] 
				min_index = v 

		return min_index 

	def primMST(self): 

		#Key values used to pick minimum weight edge in cut 
		key = [float("inf")] * self.V 
		parent = [None] * self.V # Array to store constructed MST 
		# Make key 0 so that this vertex is picked as first vertex 
		key[0] = 0
		mstSet = [False] * self.V 

		parent[0] = -1 # First node is always the root of 

		for cout in range(self.V): 
			u = self.minKey(key, mstSet) 
 
			mstSet[u] = True
			for v in range(self.V): 
				if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
						key[v] = self.graph[u][v] 
						parent[v] = u 
		return self.createMST(parent)