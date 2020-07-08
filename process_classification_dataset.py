import os, sys, h5py
import numpy as np
import pandas as pd
import subprocess


def enforce_constant_size(bed_path, output_path, window, compression=None):
    """generate a bed file where all peaks have same size centered on original peak"""

    # load bed file
    f = open(bed_path, 'rb')
    df = pd.read_table(f, header=None, compression=compression)
    print(df.head())
    #chrom = df[0].to_numpy().astype(str)
    print(dir(df[1]))
    chrom = df[0].values
    #start = df[1].to_numpy()
    start=df[1].values
    #end = df[2].to_numpy()
    end = df[2].values
    # calculate center point and create dataframe
    middle = np.round((start + end)/2).astype(int)
    half_window = np.round(int(window)/2).astype(int)

    # calculate new start and end points
    start = middle - half_window
    end = middle + half_window

    # filter any negative start positions
    data = {}
    for i in range(len(df.columns)):
        #data[i] = df[i].to_numpy()
        data[i] = df[i].values
    data[1] = start
    data[2] = end

    # create new dataframe
    df_new = pd.DataFrame(data);

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)



def parse_fasta(seq_path):
    """Parse fasta file for sequences"""

    # parse sequence and chromosome from fasta file
    num_data = np.round(sum(1 for line in open(seq_path))/2).astype(int)
    fin = open(seq_path, "r")
    sequences = []
    for j in range(num_data):
        coord = fin.readline()
        line = fin.readline()[:-1].upper()
        sequences.append(line)
    sequences = np.array(sequences)
    return sequences


def filter_nonsense_sequences(sequences):
	"""Parse fasta file for sequences"""

	# parse sequence and chromosome from fasta file
	good_index = []
	filter_sequences = []
	for i, seq in enumerate(sequences):
		if 'N' not in seq.upper():
			good_index.append(i)
			filter_sequences.append(seq)
	return np.array(filter_sequences), np.array(good_index)


def convert_one_hot(sequence, max_length=None):
	"""convert DNA/RNA sequences to a one-hot representation"""

	one_hot_seq = []
	for seq in sequence:
		seq = seq.upper()
		seq_length = len(seq)
		one_hot = np.zeros((4,seq_length))
		index = [j for j in range(seq_length) if seq[j] == 'A']
		one_hot[0,index] = 1
		index = [j for j in range(seq_length) if seq[j] == 'C']
		one_hot[1,index] = 1
		index = [j for j in range(seq_length) if seq[j] == 'G']
		one_hot[2,index] = 1
		index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
		one_hot[3,index] = 1

		# handle boundary conditions with zero-padding
		if max_length:
			offset1 = int((max_length - seq_length)/2)
			offset2 = max_length - seq_length - offset1

			if offset1:
				one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
			if offset2:
				one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

		one_hot_seq.append(one_hot)

	# convert to numpy array
	one_hot_seq = np.array(one_hot_seq)

	return one_hot_seq


def split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2):
	"""split dataset into training, cross-validation, and test set"""

	def split_index(num_data, valid_frac, test_frac):
		# split training, cross-validation, and test sets

		train_frac = 1 - valid_frac - test_frac
		cum_index = np.array(np.cumsum([0, train_frac, valid_frac, test_frac])*num_data).astype(int)
		shuffle = np.random.permutation(num_data)
		train_index = shuffle[cum_index[0]:cum_index[1]]
		valid_index = shuffle[cum_index[1]:cum_index[2]]
		test_index = shuffle[cum_index[2]:cum_index[3]]

		return train_index, valid_index, test_index


	# split training, cross-validation, and test sets
	num_data = len(one_hot)
	train_index, valid_index, test_index = split_index(num_data, valid_frac, test_frac)

	# split dataset
	train = (one_hot[train_index], labels[train_index,:])
	valid = (one_hot[valid_index], labels[valid_index,:])
	test = (one_hot[test_index], labels[test_index,:])
	indices = [train_index, valid_index, test_index]

	return train, valid, test, indices



window=sys.argv[1]
window=int(window)
experiment=sys.argv[2]
data_path = sys.argv[3]
pos_filename = sys.argv[4]
neg_filename = sys.argv[5]
genome_filename = sys.argv[6]

# set paths
#genome_path = os.path.join(data_path, genome_filename)
genome_path=genome_filename
pos_path = os.path.join(data_path, pos_filename)
neg_path = os.path.join(data_path, neg_filename)

# create new bed file with window enforced
pos_bed_path = os.path.join(data_path, experiment + '_pos_'+str(window)+'.bed')
enforce_constant_size(pos_path, pos_bed_path, window, compression='gzip')

# extract sequences from bed file and save to fasta file
pos_fasta_path = os.path.join(data_path, experiment + '_pos.fa')
cmd = ['bedtools', 'getfasta','-fi', genome_path, '-bed', pos_bed_path, '-fo', pos_fasta_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stderr)
print(stdout)

# parse sequence and chromosome from fasta file
pos_seq = parse_fasta(pos_fasta_path)

fin = open(pos_fasta_path, "r")

# filter sequences with absent nucleotides
pos_seq, _ = filter_nonsense_sequences(pos_seq)

# convert filtered sequences to one-hot representation
pos_one_hot = convert_one_hot(pos_seq, max_length=int(window))


# get non-overlap between pos peaks and neg peaks
neg_bed_path = os.path.join(data_path, experiment +'_neg.bed')
cmd = ['bedtools', 'intersect', '-v', '-wa', '-a', neg_path, '-b', pos_path, '>', neg_bed_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()


# create new bed file with window enforced
neg_bed_path2 = os.path.join(data_path, experiment + '_neg_'+str(window)+'.bed')
enforce_constant_size(neg_path, neg_bed_path2, window, compression='gzip')

# extract sequences from bed file and save to fasta file
neg_fasta_path = os.path.join(data_path, experiment + '_neg.fa')
cmd = ['bedtools', 'getfasta','-s','-fi', genome_path, '-bed', neg_bed_path2, '-fo', neg_fasta_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stderr)


# parse sequence and chromosome from fasta file
neg_seq = parse_fasta(neg_fasta_path)

# filter sequences with absent nucleotides
neg_seq, _ = filter_nonsense_sequences(neg_seq)


# convert filtered sequences to one-hot representation
neg_one_hot = convert_one_hot(neg_seq, max_length=window)



# Merge positive and negative sequences
factor = 1   # factor to balace positive and negative labelled data
num_pos = len(pos_one_hot)
num_neg = int(num_pos*factor)
match_index = np.sort(np.random.permutation(len(neg_one_hot))[:num_neg])
neg_one_hot = neg_one_hot[match_index]

# merge postive and negative sequences
one_hot = np.vstack([pos_one_hot, neg_one_hot])
labels = np.vstack([np.ones((num_pos, 1)), np.zeros((num_neg, 1))])
print(len(labels))
print('num_pos:'+str(num_pos))
print('num_neg:'+str(num_neg))

# shuffle indices for train, validation, and test sets
valid_frac = 0.1
test_frac = 0.2
train_frac = 1 - valid_frac - test_frac
num_data = len(one_hot)
cum_index = np.array(np.cumsum([0, train_frac, valid_frac, test_frac])*num_data).astype(int)
shuffle = np.random.permutation(num_data)
train_index = shuffle[cum_index[0]:cum_index[1]]
valid_index = shuffle[cum_index[1]:cum_index[2]]
test_index = shuffle[cum_index[2]:cum_index[3]]



filename = experiment+'_'+str(window)+'.h5'
file_path = os.path.join(data_path, filename)
with h5py.File(file_path, 'w') as fout:
    X_train = fout.create_dataset('x_train', data=one_hot[train_index], dtype='float32', compression="gzip")
    Y_train = fout.create_dataset('y_train', data=labels[train_index,:], dtype='int8', compression="gzip")
    X_valid = fout.create_dataset('x_valid', data=one_hot[valid_index], dtype='float32', compression="gzip")
    Y_valid = fout.create_dataset('y_valid', data=labels[valid_index,:], dtype='int8', compression="gzip")
    X_test = fout.create_dataset('x_test', data=one_hot[test_index], dtype='float32', compression="gzip")
    Y_test = fout.create_dataset('y_test', data=labels[test_index,:], dtype='int8', compression="gzip")
print('Saved to: ' + file_path)




