import os, h5py,sys
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import keras

data_path=sys.argv[1]
# output file of processing script
filename = sys.argv[2]

file_path = os.path.join(data_path, filename)
dataset = h5py.File(file_path, 'r') 
x_train = np.array(dataset['x_train']).astype(np.float32).transpose([0,2,1])
y_train = np.array(dataset['y_train']).astype(np.float32)
x_valid = np.array(dataset['x_valid']).astype(np.float32).transpose([0,2,1])
y_valid = np.array(dataset['y_valid']).astype(np.float32)
x_test = np.array(dataset['x_test']).astype(np.float32).transpose([0,2,1])
y_test = np.array(dataset['y_test']).astype(np.float32)



keras.backend.clear_session()

def conv_bn_layer(incoming, filters, kernel_size, activation, padding):
    nn = keras.layers.Conv1D(filters=filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            padding=padding,
                            )(incoming)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    return nn

def residual_block(incoming, kernel_size=5, dropout=0.1):
    num_filters = incoming.shape[-1]
    nn = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            padding='same',
                            )(incoming)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            padding='same',
                            )(nn)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([nn, incoming])
    return keras.layers.Activation('relu')(nn)

# until TF is updated comment out dilated residual block
def dilated_residual_block(incoming, kernel_size=5, dropout=0.1):
    num_filters = incoming.shape[-1]
    nn = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            dilation_rate=2,
                            padding='same',
                            )(incoming)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            dilation_rate=4,
                            padding='same',
                            )(nn)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size, 
                            activation=None, 
                            use_bias=False,
                            dilation_rate=8,
                            padding='same',
                            )(nn)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([nn, incoming])
    return keras.layers.Activation('relu')(nn)


# input layer
N, L, A = x_train.shape
inputs = keras.layers.Input(shape=(L, A))

# first layer
# revert relu back to exponential once TF updated
nn = conv_bn_layer(inputs, filters=24, kernel_size=19, 
                   activation='exponential', padding='same')


nn = keras.layers.Dropout(0.1)(nn)

# residual block
# add back in once TF is updated
nn = dilated_residual_block(nn, kernel_size=5, dropout=0.1)
nn = keras.layers.MaxPool1D(pool_size=10)(nn)
nn = keras.layers.Dropout(0.2)(nn)

# second layer
nn = conv_bn_layer(nn, filters=32, kernel_size=7, 
                   activation='relu', padding='same')
nn = keras.layers.MaxPool1D(pool_size=5)(nn)
nn = keras.layers.Dropout(0.1)(nn)

# Fully-connected NN
nn = keras.layers.Flatten()(nn)
nn = keras.layers.Dense(64, activation=None, use_bias=False)(nn)                         
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('relu')(nn)
nn = keras.layers.Dropout(0.5)(nn)

# Output layer 
logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
outputs = keras.layers.Activation('sigmoid')(logits)

# Instantiate model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# set up optimizer and metrics
auroc = keras.metrics.AUC(curve='ROC', name='auroc')
aupr = keras.metrics.AUC(curve='PR', name='aupr')
optimizer = keras.optimizers.Adam(learning_rate=0.001)
try:
    parallel_model = keras.utils.multi_gpu_model(model,gpus=1,cpu_relocation=True)
    print("Training using multiple GPUs..")
except ValueError:
    parallel_model = model
    print("Training using single GPU or CPU..")

loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
parallel_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', auroc, aupr])


#Early Stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc', #'val_aupr',#
                                            patience=20, 
                                            verbose=1, 
                                            mode='max', 
                                            restore_best_weights=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                factor=0.2,
                                                patience=5, 
                                                min_lr=1e-7,
                                                mode='max',
                                                verbose=1) 

history = parallel_model.fit(x_train, y_train, 
                    epochs=100,
                    batch_size=100, 
                    shuffle=True,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[es_callback, reduce_lr])





model_path = os.path.join(data_path, 'model_weights.hdf5')
parallel_model.save_weights(model_path)


model=parallel_model
print(model.summary())
intermediate_tmp = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
print("Intermediate tmp 1")
print(intermediate_tmp.summary())
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test results:', results)



import pandas as pd
import logomaker
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
#%matplotlib inline
#from keras import backend as K
import tensorflow.compat.v1.keras.backend as K1


def plot_filers(model, x_test, layer=3, threshold=0.5, window=20, num_cols=8, figsize=(30,5)):

    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    fmap = intermediate.predict(x_test)
    W, support = activation_pwm(fmap, x_test, threshold=threshold, window=window)

    num_filters = len(W)
    num_widths = int(np.ceil(num_filters/num_cols))

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    logos = []
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_widths, num_cols, n+1)
        #if (np.sum(w) != 0) | (np.sum(np.isnan(w) == True) > 0):
        
        # calculate sequence logo heights
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w

        L, A = w.shape
        counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(L)))
        for a in range(A):
            for l in range(L):
                counts_df.iloc[l,a] = logo[l,a]

        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
        
        logos.append(logo)
        
    return fig, W, logo



def activation_pwm(fmap, X, threshold=0.5, window=20):

    # extract sequences with aligned activation
    window_left = int(window/2)
    window_right = window - window_left

    N,seq_length,num_dims = X.shape
    num_filters = fmap.shape[-1]

    support = []
    W = []
    for filter_index in range(num_filters):

        # find regions above threshold
        try:
            coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)
            x, y = coords

            # sort score
            index = np.argsort(fmap[x,y,filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            support.append(len(pos_index))

            seq_align = []
            count_matrix = []
            for i in range(len(pos_index)):

                # handle boundary conditions at start
                start_window = pos_index[i] - window_left
                if start_window < 0:
                    start_buffer = np.zeros((-start_window, num_dims))
                    start = 0
                else:
                    start = start_window

                # handle boundary conditions at end 
                end_window = pos_index[i] + window_right
                end_remainder = end_window - seq_length
                if end_remainder > 0:
                    end = seq_length
                    end_buffer = np.zeros((end_remainder, num_dims))
                else:
                    end = end_window

                seq = X[data_index[i],start:end,:]

                if start_window < 0:
                    seq = np.vstack([start_buffer, seq])
                if end_remainder > 0:
                    seq = np.vstack([seq, end_buffer])

                seq_align.append(seq)
                #weight = fmap[data_index[i],pos_index[i],filter_index]
                #seq_align.append(seq*weight)
                #count_matrix.append(np.sum(seq, axis=1, keepdims=True)*weight)

            #seq_align = np.array(seq_align)
            #count_matrix = np.array(count_matrix)

            # normalize counts
            #seq_align = np.sum(seq_align, axis=0)/np.sum(count_matrix, axis=0)*np.ones((window,4))
            #seq_align[np.isnan(seq_align)] = 0
            #seq_align[seq_align < 0] = 0
                
            seq_align = np.array(seq_align)
            W.append(np.mean(seq_align, axis=0))
        except:
            W.append(np.ones((window,4))/4)
    W = np.array(W)

    return W, support


def saliency(model, X, class_index=0, layer=-2, batch_size=256):
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]
    sess = K1.get_session()

    N = len(X)
    num_batches = int(np.floor(N/batch_size))

    attr_score = []
    for i in range(num_batches):
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[i*batch_size:(i+1)*batch_size]}))
    if num_batches*batch_size < N:
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[num_batches*batch_size:N]}))

    return np.concatenate(attr_score, axis=0)


def mutagenesis(model, X, class_index=0, layer=-2):

    def generate_mutagenesis(X):
        L,A = X.shape 

        X_mut = []
        for l in range(L):
            for a in range(A):
                X_new = np.copy(X)
                X_new[l,:] = 0
                X_new[l,a] = 1
                X_mut.append(X_new)
        return np.array(X_mut)

    N, L, A = X.shape 
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

    attr_score = []
    for x in X:

        # get baseline wildtype score
        wt_score = intermediate.predict(np.expand_dims(x, axis=0))[:, class_index]

        # generate mutagenized sequences
        x_mut = generate_mutagenesis(x)
        
        # get predictions of mutagenized sequences
        predictions = intermediate.predict(x_mut)[:,class_index]

        # reshape mutagenesis predictiosn
        mut_score = np.zeros((L,A))
        k = 0
        for l in range(L):
            for a in range(A):
                mut_score[l,a] = predictions[k]
                k += 1
                
        attr_score.append(mut_score - wt_score)
    return np.array(attr_score)


def clip_filters(W, threshold=0.5, pad=3):

    W_clipped = []
    for w in W:
        L,A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped



def meme_generate(W, output_file='meme.txt', prefix='filter'):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
        for i in range(L):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
        f.write('\n')

    f.close()



num_plots = 10
class_index = 0

# get high predicted sequences 
pos_index = np.where(y_test[:, class_index] == 1)[0]
predictions = model.predict(x_test[pos_index])
plot_index = pos_index[np.argsort(predictions[:,class_index])[::-1]]
X = x_test[plot_index[0:num_plots]]

# get attribution scores
attr_score = saliency(model, X, class_index=0, layer=-2, batch_size=256)
attr_score = attr_score * X


# plot attribution scores for sequences with top predictions
N, L, A = attr_score.shape
for i in range(len(X)):
    counts_df = pd.DataFrame(data=0.0, columns=list('ACGU'), index=list(range(L)))
    for a in range(A):
        for l in range(L):
            counts_df.iloc[l,a] = attr_score[i][l,a]

    logomaker.Logo(counts_df, figsize=(25,2))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()


fig, W, logo = plot_filers(model, x_test, layer=3, threshold=0.5, 
                                    window=20, num_cols=8, figsize=(30,5))

# generate meme file
W_clipped = clip_filters(W, threshold=0.5, pad=3)
output_file = os.path.join(data_path, 'filters.meme')
print("output_file: ",output_file)
meme_generate(W_clipped, output_file) 



#!tomtom -evalue -thresh 0.1 -o ../data/filters ../data/filters.meme ../data/JASPAR_CORE_2016_vertebrates.meme

