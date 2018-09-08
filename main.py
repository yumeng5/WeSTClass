import numpy as np
np.random.seed(1234)
from time import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model import WSTC, f1
from keras.optimizers import SGD
from gen import augment, pseudodocs
from load_data import load_dataset
from gensim.models import word2vec


def train_word2vec(sentence_matrix, vocabulary_inv, dataset_name, mode='skipgram',
                   num_features=100, min_word_count=5, context=5):
    model_dir = './' + dataset_name
    model_name = "embedding"
    model_name = os.path.join(model_dir, model_name)
    if os.path.exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Loading existing Word2Vec model {}...".format(model_name))
    else:
        num_workers = 15  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')

        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        if mode == 'skipgram':
            sg = 1
            print('Model: skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model: CBOW')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, sg=sg,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        embedding_model.init_sims(replace=True)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Saving Word2Vec model {}".format(model_name))
        embedding_model.save(model_name)

    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                        np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights


def write_output(write_path, y_pred, perm):
    invperm = np.zeros(len(perm), dtype='int32')
    for i,v in enumerate(perm):
        invperm[v] = i
    y_pred = y_pred[invperm]
    with open(os.path.join(write_path, 'out.txt'), 'w') as f:
        for val in y_pred:
            f.write(str(val) + '\n')
    print("Classification results are written in {}".format(os.path.join(write_path, 'out.txt')))
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ### Basic settings ###
    # dataset selection: AG's News (default) and Yelp Review
    parser.add_argument('--dataset', default='agnews', choices=['agnews', 'yelp'])
    # neural model selection: Convolutional Neural Network (default) and Hierarchical Attention Network
    parser.add_argument('--model', default='cnn', choices=['cnn', 'rnn'])
    # weak supervision selection: label surface names (default), class-related keywords and labeled documents
    parser.add_argument('--sup_source', default='labels', choices=['labels', 'keywords', 'docs'])
    # whether ground truth labels are available for evaluation: True (default), False
    parser.add_argument('--with_evaluation', default='True', choices=['True', 'False'])

    ### Training settings ###
    # mini-batch size for both pre-training and self-training: 256 (default)
    parser.add_argument('--batch_size', default=256, type=int)
    # maximum self-training iterations: 5000 (default)
    parser.add_argument('--maxiter', default=5e3, type=int)
    # pre-training epochs: None (default)
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    # self-training update interval: None (default)
    parser.add_argument('--update_interval', default=None, type=int)

    ### Hyperparameters settings ###
    # background word distribution weight (alpha): 0.2 (default)
    parser.add_argument('--alpha', default=0.2, type=float)
    # number of generated pseudo documents per class (beta): 500 (default)
    parser.add_argument('--beta', default=500, type=int)
    # keyword vocabulary size (gamma): 50 (default)
    parser.add_argument('--gamma', default=50, type=int)
    # self-training stopping criterion (delta): None (default)
    parser.add_argument('--delta', default=0.1, type=float)

    ### Case study settings ###
    # trained model directory: None (default)
    parser.add_argument('--trained_weights', default=None)

    args = parser.parse_args()
    print(args)

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta = args.delta

    word_embedding_dim = 100
    
    if args.model == 'cnn':

        if args.dataset == 'agnews':
            update_interval = 50
            pretrain_epochs = 20
            self_lr = 1e-3
            max_sequence_length = 100

        elif args.dataset == 'yelp':
            update_interval = 50
            pretrain_epochs = 30
            self_lr = 1e-4
            max_sequence_length = 500

        decay = 1e-6
    
    elif args.model == 'rnn':

        if args.dataset == 'agnews':
            update_interval = 50
            pretrain_epochs = 100
            self_lr = 1e-3
            sent_len = 45
            doc_len = 10

        elif args.dataset == 'yelp':
            update_interval = 100
            pretrain_epochs = 200
            self_lr = 1e-4
            sent_len = 30
            doc_len = 40

        decay = 1e-5
        max_sequence_length = [doc_len, sent_len]

    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    if args.with_evaluation == 'True':
        with_evaluation = True
    else:
        with_evaluation = False
    if args.sup_source == 'labels' or args.sup_source == 'keywords':
        x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, perm = \
            load_dataset(args.dataset, model=args.model, sup_source=args.sup_source, with_evaluation=with_evaluation, truncate_len=max_sequence_length)
        sup_idx = None
    elif args.sup_source == 'docs':
        x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, sup_idx, perm = \
            load_dataset(args.dataset, model=args.model, sup_source=args.sup_source, with_evaluation=with_evaluation, truncate_len=max_sequence_length)
    
    np.random.seed(1234)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    vocab_sz = len(vocabulary_inv)
    n_classes = len(word_sup_list)    

    if args.model == 'cnn':
        if x.shape[1] < max_sequence_length:
            max_sequence_length = x.shape[1]
        x = x[:, :max_sequence_length]
        sequence_length = max_sequence_length

    elif args.model == 'rnn':
        if x.shape[1] < doc_len:
            doc_len = x.shape[1]
        if x.shape[2] < sent_len:
            sent_len = x.shape[2]
        x = x[:, :doc_len, :sent_len]
        sequence_length = [doc_len, sent_len]
    
    print("\n### Input preparation ###")
    embedding_weights = train_word2vec(x, vocabulary_inv, args.dataset)
    embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocabulary_inv])
    
    wstc = WSTC(input_shape=x.shape, n_classes=n_classes, y=y, model=args.model,
                vocab_sz=vocab_sz, embedding_matrix=embedding_mat, word_embedding_dim=word_embedding_dim)

    if args.trained_weights is None:
        print("\n### Phase 1: vMF distribution fitting & pseudo document generation ###")
        
        word_sup_array = np.array([np.array([vocabulary[word] for word in word_class_list]) for word_class_list in word_sup_list])
        
        total_counts = sum(word_counts[ele] for ele in word_counts)
        total_counts -= word_counts[vocabulary_inv_list[0]]
        background_array = np.zeros(vocab_sz)
        for i in range(1,vocab_sz):
            background_array[i] = word_counts[vocabulary_inv[i]]/total_counts
        seed_docs, seed_label = pseudodocs(word_sup_array, gamma, background_array,
                                           sequence_length, len_avg, len_std, beta, alpha, 
                                           vocabulary_inv, embedding_mat, args.model, 
                                           './results/{}/{}/phase1/'.format(args.dataset, args.model))
        
        if args.sup_source == 'docs':
            if args.model == 'cnn':
                num_real_doc = len(sup_idx.flatten()) * 10
            elif args.model == 'rnn':
                num_real_doc = len(sup_idx.flatten())
            real_seed_docs, real_seed_label = augment(x, sup_idx, num_real_doc)
            seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
            seed_label = np.concatenate((seed_label, real_seed_label), axis=0)

        perm_seed = np.random.permutation(len(seed_label))
        seed_docs = seed_docs[perm_seed]
        seed_label = seed_label[perm_seed]

        print('\n### Phase 2: pre-training with pseudo documents ###')

        wstc.pretrain(x=seed_docs, pretrain_labels=seed_label,
                     sup_idx=sup_idx, optimizer=SGD(lr=0.1, momentum=0.9),
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     save_dir='./results/{}/{}/phase2'.format(args.dataset, args.model))

        y_pred = wstc.predict(x)
        if y is not None:
            f1_macro, f1_micro = np.round(f1(y, y_pred), 5)
            print('F1 score after pre-training: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))

        t0 = time()
        print("\n### Phase 3: self-training ###")
        selftrain_optimizer = SGD(lr=self_lr, momentum=0.9, decay=decay)
        wstc.compile(optimizer=selftrain_optimizer, loss='kld')
        y_pred = wstc.fit(x, y=y, tol=delta, maxiter=args.maxiter, batch_size=args.batch_size,
                         update_interval=update_interval, save_dir='./results/{}/{}/phase3'.format(args.dataset, args.model), 
                         save_suffix=args.dataset+'_'+str(args.sup_source))
        print('Self-training time: {:.2f}s'.format(time() - t0))

    else:
        print("\n### Directly loading trained weights ###")
        wstc.load_weights(args.trained_weights)
        y_pred = wstc.predict(x)
        if y is not None:
            f1_macro, f1_micro = np.round(f1(y, y_pred), 5)
            print('F1 score: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))
    
    print("\n### Generating outputs ###")
    write_output('./' + args.dataset, y_pred, perm)
