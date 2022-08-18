import matplotlib.pyplot as plt
import reader as rd

def visualize_clustme():
	clustme_data = rd.read_clustme_data()

	subplot_num = 10
	block_size  = subplot_num ** 2
	block_num   = int(len(clustme_data) / block_size)

	cmap = plt.get_cmap('coolwarm')
	for i in range(block_num):
		fig, axs = plt.subplots(subplot_num, subplot_num)
		fig.set_facecolor('black')
		fig.set_size_inches(60, 60)
		for j in range(subplot_num):
			for k in range(subplot_num):
				axs[j][k].scatter(
					clustme_data[i * block_size + j * subplot_num + k]["data"][:, 0], 
					clustme_data[i * block_size + j * subplot_num + k]["data"][:, 1],
					c=cmap(clustme_data[i * block_size + j * subplot_num + k]["prob_single"])
				)
				axs[j][k].axis("off")
				
	
	  ## save the current block figure
		fig.savefig("./plot/clustme_" + str(i) + ".png")
		
		plt.clf()

visualize_clustme()