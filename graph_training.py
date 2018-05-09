import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams

csv_dir = './nmt/training_stats/'

train_runs = ['Baseline.csv', 'Skip 5.csv', 'Skip 10.csv']


def set_plot_formats():
	rc('axes', linewidth=4)
	rc('xtick.major', pad=13)
	rc('ytick.major', pad=10)
set_plot_formats()

def set_ax_formats(ax):
	ax.get_yaxis().tick_left()
	ax.get_xaxis().tick_bottom()
	ax.tick_params(axis='both', which='major', labelsize=45, length=15, width=6)
	ax.tick_params(axis='both', which='minor', length=8, width=3)
	ax.legend(prop={'size':30})



fig = plt.figure(1,figsize=(15,15))
ax = fig.add_subplot(1,1,1)

for csv_file in train_runs:
	print(csv_file)
	arr = np.loadtxt(csv_dir+csv_file, delimiter=',', skiprows=1)
	steps = arr[:,0]
	ppl = arr[:,1]
	gN = arr[:,2]
	bleu = arr[:,3]
	ax.plot(steps, bleu, label=csv_file[:-4], linewidth=6)
	# ax.set_yscale('log')

	ax.set_title("Best BLEU Through\n 20k Training Steps", fontsize=70, pad=40)
	ax.set_xlabel("Step", fontsize=60, labelpad=30)
	ax.set_ylabel("BLEU", fontsize=60, labelpad=30)

set_ax_formats(ax)
plt.gcf().subplots_adjust(left=.18, bottom=.16, top=.8)
f = open("train_bleu_vs_step_v3.png", 'wb+')
fig.savefig(f, format='png', dpi=300)