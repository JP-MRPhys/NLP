def load_glove():
    glove_filename = '/home/jehill/python/NLP/datasets/GloVE/glove.6B.300d.txt'

    glove_vocab = []
    glove_embed = []
    embedding_dict = {}

    file = open(glove_filename, 'r', encoding='UTF-8')

    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]]  # convert to list of float
        embedding_dict[vocab_word] = embed_vector
        print(embed_vector)
        glove_embed.append(embed_vector)

    print('Loaded GLOVE')
    file.close()

    return glove_vocab, glove_embed, embedding_dict






if __name__ == '__main__':
    # glove_filename = '/home/jehill/python/NLP/datasets/GloVE/glove.6B.300d.txt'
    # dir = '/home/jehill/python/NLP/nmt-master/nmt/wmt16_en_de_preprocessed/'

    # glove_vocab, glove_embed, embedding_dict = load_glove()
    #encoder_emb = word_embedding_matrix(source_vocab_file, source_vocab, dim=300)

    """    
    # look up our word vectors and store them as numpy arrays
    
    

    king_vector = np.array(embedding_dict['king'])
    man_vector = np.array(embedding_dict['man'])
    woman_vector = np.array(embedding_dict['woman'])

    # add/subtract our vectors

    new_vector = king_vector - man_vector + woman_vector

    print(king_vector.shape)

    # here we use a scipy function to create a "tree" of word vectors
    # that we can run queries against

    tree = spatial.KDTree(glove_embed)

    # run query with our new_vector to find the closest word vectors

    nearest_dist, nearest_idx = tree.query(new_vector, 10)
    nearest_words = [glove_vocab[i] for i in nearest_idx]
    print(nearest_words)

    ['king', 'queen', 'monarch', 'mother', 'princess', 'daughter', 'elizabeth', 'throne', 'kingdom', 'wife']
    """
