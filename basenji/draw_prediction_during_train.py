import os
import sys
import pickle
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
sys.path.append(source_path)
sys.path.append(source_path2)
import json
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' ### run on CPU

import tensorflow as tf
print(tf.__version__)
if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

import numpy as np
import pandas as pd
import pysam
import matplotlib.pyplot as plt
from cooltools.lib.numutils import set_diag
import dataset, dna_io, seqnn

# model_dir = './explore_best_model/'
def draw_pred_in_train(model,out_dir, epoch, loss, pearson_r):
    model_dir = out_dir
    fasta_file = "/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/genomes/AalbS2_V4.fa"
    params_file = model_dir+'/params.json'
    # model_file  = model_dir+'model_best.h5'
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']
    seqnn_model = model
    ### names of targets ###
    data_dir ='/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048'
    # data_dir ='/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aste_2048_globaloe'
    hic_targets = pd.read_csv(data_dir+'/targets.txt',sep='\t')
    hic_file_dict_num = dict(zip(hic_targets['index'].values, hic_targets['file'].values) )
    hic_file_dict     = dict(zip(hic_targets['identifier'].values, hic_targets['file'].values) )
    hic_num_to_name_dict = dict(zip(hic_targets['index'].values, hic_targets['identifier'].values) )

    # read data parameters
    data_stats_file = '%s/statistics.json' % data_dir
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    hic_diags =  data_stats['diagonal_offset']
    target_crop = data_stats['crop_bp'] // data_stats['pool_width']
    target_length1 = data_stats['seq_length'] // data_stats['pool_width']

    ### load data ###
    sequences = pd.read_csv(data_dir+'/sequences.bed', sep='\t', names=['chr','start','stop','type'])
    sequences_test = sequences.iloc[sequences['type'].values=='test']
    sequences_test.reset_index(inplace=True, drop=True)
    #print("going to load test dataset")
    test_data = dataset.SeqDataset(data_dir, 'test', batch_size=8)

    # test_targets is a float array with shape
    # [#regions, #pixels, target #target datasets]
    # representing log(obs/exp)data, where #pixels
    # corresponds to the number of entries in the flattened
    # upper-triangular representation of the matrix

    # test_inputs are 1-hot encoded arrays with shape
    # [#regions, 2^20 bp, 4 nucleotides datasets]

    test_inputs, test_targets = test_data.numpy(return_inputs=True, return_outputs=True)
    # print(test_targets)

    ### for converting from flattened upper-triangluar vector to symmetric matrix  ###
    def from_upper_triu(vector_repr, matrix_len, num_diags):
        z = np.zeros((matrix_len,matrix_len))
        triu_tup = np.triu_indices(matrix_len,num_diags)
        z[triu_tup] = vector_repr
        for i in range(-num_diags+1,num_diags):
            set_diag(z, np.nan, i)
        return z + z.T

    target_length1_cropped = target_length1 - 2*target_crop
    #print('flattened representation length:', target_length)
    #print('symmetrix matrix size:', '('+str(target_length1_cropped)+','+str(target_length1_cropped)+')')

    fig2_examples = [
                        #Aalb
                        # '2L:36888576-37937152',
                        # '2L:24403968-25452544',
                        '2L:12992512-14041088',
                        # '2R:40747008-41795584',
                        # '3R:26742784-27791360',
                        # '3R:25497600-26546176',
                        # '3R:25530368-26578944',
                        # '3R:25563136-26611712',
                        # '3R:25595904-6644480',
                        # '3R:25628672-26677248',
                        # '3R:26939392-27987968',
                        # 'X:2482176-3530752'
                        #Aste
                        # '2R:32083968-33132544',
                        # '2R:32116736-33165312',

                        ]
                        # 'chr11:75429888-76478464',
                        # 'chr15:63281152-64329728'

    fig2_inds = []
    for seq in fig2_examples:
        #print(seq)
        # print(np.unique(sequences_test['chr'].values))
        chrm,start,stop = seq.split(':')[0], seq.split(':')[1].split('-')[0], seq.split(':')[1].split('-')[1]
        # print(np.where(sequences_test['chr'].values== chrm))
        # print(np.where(sequences_test['start'].values== int(start)))
        # print(np.where(sequences_test['stop'].values==  int(stop )))
        test_ind = np.where( (sequences_test['chr'].values== chrm) *
                             (sequences_test['start'].values== int(start))*
                             (sequences_test['stop'].values==  int(stop ))  )[0][0]
        fig2_inds.append(test_ind)
    #print(fig2_inds)

    ### make predictions and plot the three examples above ###

    target_index = 0  #Aalb
    for test_index in fig2_inds:
        chrm, seq_start, seq_end = sequences_test.iloc[test_index][0:3]
        myseq_str = chrm + ':' + str(seq_start) + '-' + str(seq_end)
        #print(' ')
        #print(myseq_str)

        test_target = test_targets[test_index:test_index + 1, :, :]
        fasta_open = pysam.Fastafile(fasta_file)
        seq = fasta_open.fetch(chrm, seq_start, seq_end).upper()
        # with open(data_dir+"/prediction/" + "exploreseq" + str(seq_start) + "-" + str(seq_end) + ".pickle", 'wb') as f:
        #     pickle.dump(seq, f)
        # print(seq)
        seq_1hot = dna_io.dna_1hot(seq)
        # print(seq[21680:21687])
        # print(seq_1hot[21680:21687][:])

        # with open(data_dir+"/prediction/" + "explore" + str(seq_start) + "-" + str(seq_end) + ".pickle", 'wb') as f:
        #     pickle.dump(seq_1hot, f)
        test_pred  = seqnn_model.model.predict(np.expand_dims(seq_1hot, 0))
        # test_pred = seqnn_model.model.predict(test_inputs[test_index:test_index + 1, :, :])
        # print("inputs")
        # print(test_inputs[test_index:test_index + 1, :, :])
        # print(seq_1hot)


        plt.figure(figsize=(8, 4))
        target_index = 0
        vmin = -2
        vmax = 2

        # plot pred

        mat = from_upper_triu(test_pred[:, :, target_index], target_length1_cropped, hic_diags)
        #print(target_index, target_length1_cropped, hic_diags)
        #print(mat)
        plt.subplot(121)
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r')#, vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05)#, ticks=[-2, -1, 0, 1, 2])
        plt.title('pred-' + str(hic_num_to_name_dict[target_index]+myseq_str), y=1.15)
        plt.ylabel(myseq_str)
        # plt.clf()
        # plot target
        plt.subplot(122)
        mat = from_upper_triu(test_target[:, :, target_index], target_length1_cropped, hic_diags)
        # print(mat)
        # print(np.max(mat))
        # print(np.min(mat))
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r')#, vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05)#, ticks=[-2, -1, 0, 1, 2])
        plt.title('target-' + str(hic_num_to_name_dict[target_index]+myseq_str), y=1.15)
        plt.tight_layout()
        plt.suptitle("epoch " + str(epoch) + "; valid loss " + str(loss) + "; valid Pearson "+str(pearson_r))
        plt.savefig(model_dir+"/prediction_"+str(chrm)+"_"+str(seq_start)+"_"+str(seq_end)+"epoch"+str(epoch)+".png")
        plt.clf()