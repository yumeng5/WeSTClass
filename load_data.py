import csv
import numpy as np
import os
import re
import itertools
import pickle
from collections import Counter
from os.path import join
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def read_file(data_dir, with_evaluation):
    data = []
    target = []
    with open(join(data_dir, 'dataset.csv'), 'rt', encoding='utf-8') as csvfile:
        csv.field_size_limit(500 * 1024 * 1024)
        reader = csv.reader(csvfile)
        for row in reader:
            if data_dir == './agnews':
                doc = row[1] + '. ' + row[2]
                data.append(doc)
                target.append(int(row[0]) - 1)
            elif data_dir == './yelp':
                data.append(row[1])
                target.append(int(row[0]) - 1)
    if with_evaluation:
        y = np.asarray(target)
        assert len(data) == len(y)
        assert set(range(len(np.unique(y)))) == set(np.unique(y))
    else:
        y = None
    return data, y


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_doc(data):
    data = [s.strip() for s in data]
    data = [clean_str(s) for s in data]
    return data


def pad_sequences(sentences, padding_word="<PAD/>", pad_len=None):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)
    
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv


def build_input_data_cnn(sentences, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def build_input_data_rnn(data, vocabulary, max_doc_len, max_sent_len):
    x = np.zeros((len(data), max_doc_len, max_sent_len), dtype='int32')
    for i, doc in enumerate(data):
        for j, sent in enumerate(doc):
            k = 0
            for word in sent:
                x[i,j,k] = vocabulary[word]
                k += 1         
    return x


def extract_keywords(data_path, vocab, class_type, num_keywords, data, perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, doc_ids = line.split(':')
        assert int(class_id) == i
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            sup_data.append(" ".join(data[idx]))
            sup_label.append(i)

    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk

    tfidf = TfidfVectorizer(norm='l2', sublinear_tf=True, max_df=0.2, stop_words='english')
    sup_x = tfidf.fit_transform(sup_data)
    sup_x = np.asarray(sup_x.todense())

    vocab_dict = tfidf.vocabulary_
    vocab_inv_dict = {v: k for k, v in vocab_dict.items()}

    print("\n### Supervision type: Labeled documents ###")
    print("Extracted keywords for each class: ")
    keywords = []
    cnt = 0
    for i in range(len(sup_idx)):
        class_vec = np.average(sup_x[cnt:cnt+len(sup_idx[i])], axis=0)
        cnt += len(sup_idx[i])
        sort_idx = np.argsort(class_vec)[::-1]
        keyword = []
        if class_type == 'topic':
            j = 0
            k = 0
            while j < num_keywords:
                w = vocab_inv_dict[sort_idx[k]]
                if w in vocab:
                    keyword.append(vocab_inv_dict[sort_idx[k]])
                    j += 1
                k += 1
        elif class_type == 'sentiment':
            j = 0
            k = 0
            while j < num_keywords:
                w = vocab_inv_dict[sort_idx[k]]
                w, t = nltk.pos_tag([w])[0]
                if t.startswith("J") and w in vocab:
                    keyword.append(w)
                    j += 1
                k += 1
        print("Class {}:".format(i))
        print(keyword)
        keywords.append(keyword)

    new_sup_idx = []
    m = {v: k for k, v in enumerate(perm)}
    for seed_idx in sup_idx:
        new_seed_idx = []
        for ele in seed_idx:
            new_seed_idx.append(m[ele])
        new_sup_idx.append(new_seed_idx)
    new_sup_idx = np.asarray(new_sup_idx)

    return keywords, new_sup_idx


def load_keywords(data_path, sup_source):
    if sup_source == 'labels':
        file_name = 'classes.txt'
        print("\n### Supervision type: Label Surface Names ###")
        print("Label Names for each class: ")
    elif sup_source == 'keywords':
        file_name = 'keywords.txt'
        print("\n### Supervision type: Class-related Keywords ###")
        print("Keywords for each class: ")
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    
    keywords = []
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, contents = line.split(':')
        assert int(class_id) == i
        keyword = contents.split(',')
        print("Supervision content of class {}:".format(i))
        print(keyword)
        keywords.append(keyword)
    return keywords


def load_cnn(dataset_name, sup_source, num_keywords=10, with_evaluation=True, truncate_len=None):
    data_path = './' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    sz = len(data)
    np.random.seed(1234)
    perm = np.random.permutation(sz)

    data = preprocess_doc(data)
    data = [s.split(" ") for s in data]

    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics: ###")
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))

    if truncate_len is None:
        truncate_len = min(int(len_avg + 3*len_std), len_max)
    print("Defined maximum document length: {} (words)".format(truncate_len))
    print('Fraction of truncated documents: {}'.format(sum(tmp > truncate_len for tmp in tmp_list)/len(tmp_list)))
    
    sequences_padded = pad_sequences(data)
    word_counts, vocabulary, vocabulary_inv = build_vocab(sequences_padded)
    x = build_input_data_cnn(sequences_padded, vocabulary)
    x = x[perm]

    if with_evaluation:
        print("Number of classes: {}".format(len(np.unique(y))))
        print("Number of documents in each class:")
        for i in range(len(np.unique(y))):
            print("Class {}: {}".format(i, len(np.where(y == i)[0])))
        y = y[perm]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    if sup_source == 'labels' or sup_source == 'keywords':
        keywords = load_keywords(data_path, sup_source)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, len_std, keywords, perm
    elif sup_source == 'docs':
        if dataset_name == 'nyt':
            class_type = 'topic'
        elif dataset_name == 'agnews':
            class_type = 'topic'
        elif dataset_name == 'yelp':
            class_type = 'sentiment'
        keywords, sup_idx = extract_keywords(data_path, vocabulary, class_type, num_keywords, data, perm)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, len_std, keywords, sup_idx, perm


def load_rnn(dataset_name, sup_source, num_keywords=10, with_evaluation=True, truncate_len=None):
    data_path = './' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    sz = len(data)
    np.random.seed(1234)
    perm = np.random.permutation(sz)

    data = preprocess_doc(data)
    data_copy = [s.split(" ") for s in data]
    docs_padded = pad_sequences(data_copy)
    word_counts, vocabulary, vocabulary_inv = build_vocab(docs_padded)

    data = [tokenize.sent_tokenize(doc) for doc in data]
    flat_data = [sent for doc in data for sent in doc]

    tmp_list = [len(sent.split(" ")) for sent in flat_data]
    max_sent_len = max(tmp_list)
    avg_sent_len = np.average(tmp_list)
    std_sent_len = np.std(tmp_list)

    print("\n### Dataset statistics: ###")
    print('Sentence max length: {} (words)'.format(max_sent_len))
    print('Sentence average length: {} (words)'.format(avg_sent_len))
    
    if truncate_len is None:
        truncate_sent_len = min(int(avg_sent_len + 3*std_sent_len), max_sent_len)
    else:
        truncate_sent_len = truncate_len[1]
    print("Defined maximum sentence length: {} (words)".format(truncate_sent_len))
    print('Fraction of truncated sentences: {}'.format(sum(tmp > truncate_sent_len for tmp in tmp_list)/len(tmp_list)))

    tmp_list = [len(doc) for doc in data]
    max_doc_len = max(tmp_list)
    avg_doc_len = np.average(tmp_list)
    std_doc_len = np.std(tmp_list)
    
    print('Document max length: {} (sentences)'.format(max_doc_len))
    print('Document average length: {} (sentences)'.format(avg_doc_len))

    if truncate_len is None:
        truncate_doc_len = min(int(avg_doc_len + 3*std_doc_len), max_doc_len)
    else:
        truncate_doc_len = truncate_len[0]
    print("Defined maximum document length: {} (sentences)".format(truncate_doc_len))
    print('Fraction of truncated documents: {}'.format(sum(tmp > truncate_doc_len for tmp in tmp_list)/len(tmp_list)))
    
    len_avg = [avg_doc_len, avg_sent_len]
    len_std = [std_doc_len, std_sent_len]

    data = [[sent.split(" ") for sent in doc] for doc in data]
    x = build_input_data_rnn(data, vocabulary, max_doc_len, max_sent_len)
    x = x[perm]

    if with_evaluation:
        print("Number of classes: {}".format(len(np.unique(y))))
        print("Number of documents in each class:")
        for i in range(len(np.unique(y))):
            print("Class {}: {}".format(i, len(np.where(y == i)[0])))
        y = y[perm]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    if sup_source == 'labels' or sup_source == 'keywords':
        keywords = load_keywords(data_path, sup_source)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, len_std, keywords, perm
    elif sup_source == 'docs':
        if dataset_name == 'nyt':
            class_type = 'topic'
        elif dataset_name == 'agnews':
            class_type = 'topic'
        elif dataset_name == 'yelp':
            class_type = 'sentiment'
        keywords, sup_idx = extract_keywords(data_path, vocabulary, class_type, num_keywords, data_copy, perm)
        return x, y, word_counts, vocabulary, vocabulary_inv, len_avg, len_std, keywords, sup_idx, perm


def load_dataset(dataset_name, sup_source, model='cnn', with_evaluation=True, truncate_len=None):
    if model == 'cnn':
        return load_cnn(dataset_name, sup_source, with_evaluation=with_evaluation, truncate_len=truncate_len)
    elif model == 'rnn':
        return load_rnn(dataset_name, sup_source, with_evaluation=with_evaluation, truncate_len=truncate_len)
