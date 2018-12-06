# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9

import json
import os

import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
from sklearn.utils import shuffle


def download_coco():
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)

    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
    return annotation_file


def get_coco_datasets(number=30000):
    annotation_file = './annotations/captions_train2014.json'
    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):

        print("Training dataset set not found downloading dataset")
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + '/train2014/'
    else:
        PATH = os.path.abspath('.') + '/train2014/'

    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffling the captions and image_names together
    # setting a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    # selecting the first "number" of captions from the shuffled set
    num_examples = number
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return train_captions, img_name_vector


def read_images(filename, dim1, dim2):
    image = ndimage.imread(filename, mode="RGB")
    image_resized = misc.imresize(image, (dim1, dim2))

    return image_resized


def get_caption_word2vec(batch_captions, max_sequence_lenght, embedding_dict):
    """

    :param batch_captions:
    :return: a numpy array of for word2vec of caption i.e. [batch_size, max_sequence_length,embeddimg_dim]
    """

    for caption in batch_captions:
        for token in caption.split():
            embedding_vector = np.array(embedding_dict[token])

    return


def load_glove():
    print('loading glove embeddings.. takes 2 mins generally')

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
        glove_embed.append(embed_vector)

    print('Completed loading glove embeddings..')
    file.close()

    return glove_vocab, glove_embed, embedding_dict


def load_glove_small():
    print('loading glove embeddings.. takes 2 mins generally')

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
        glove_embed.append(embed_vector)

    print('Completed loading glove embeddings..')
    file.close()

    return glove_vocab, glove_embed, embedding_dict




def get_embedding_vector(captions, embedding_dict, max_seq_length):
    """
    
    :param captions: these are input captions 
    :param embedding_dict: these embedding vectors dictonary
    :param max_seq_length: is the max_seq_length
    :return: embedding_vectors i.e. word2vec array for captions of [max_seq_length, embedding_dim]
    """  # returns

    count = 0
    unknown_token_embedding = embedding_dict['unk']
    pad_token_embedding = embedding_dict['pad']
    embedding_dim = np.shape(pad_token_embedding)[0]
    embedding_vector = np.zeros([max_seq_length, embedding_dim])
    tokens = captions.split()

    tokens = tokens[1:]  # remove <start>
    tokens = tokens[:-1]  # remove <end>

    for count in range(max_seq_length):

        if count < len(tokens):
            if tokens[count].lower() in embedding_dict:
                embedding_vector[count, :] = embedding_dict[tokens[count].lower()]

            else:
                # print("Unknown token " + token.lower())
                embedding_vector[count, :] = unknown_token_embedding

        else:
            # pad the embedding vector
            embedding_vector[count, :] = pad_token_embedding
            count = count + 1

    return embedding_vector


def get_batch_data_image_caption(imagefiles, captions, word_embedding_dict, max_sequence_length):
    """
    Read the images and caption to

    :param image_filenames:
    :param captions:
    :param word2vec_emdedding dict: is the dictonary containing word to vectors (e.g. glove, muse) etc
    :return: images_array and wordvec for captions
    """

    dim1 = 244
    dim2 = 244
    dim3 = 3
    embedding_dim = np.shape(word_embedding_dict['pad'])[0]  # 300 for glove embedding

    batch_images = np.zeros([len(imagefiles), dim1, dim2, dim3])
    batch_caption_vectors = np.zeros([len(imagefiles), max_sequence_length, embedding_dim])

    for i in range(len(imagefiles)):
        batch_images[i, :, :, :] = read_images(imagefiles[i], dim1, dim2)
        batch_caption_vectors[i, :, :] = get_embedding_vector(captions[i], word_embedding_dict, max_sequence_length)

    return batch_images, batch_caption_vectors


if __name__ == '__main__':

    BATCH_SIZE = 50
    glove_vocab, glove_embed, word_embedding_dict = load_glove()

    print("Glove vocab shape" + str(len(glove_vocab)))
    print("Glove embedding shape:")
    print(np.shape(glove_embed)[0])

    captions, image_filename = get_coco_datasets();

    print("Number of captions" + str(len(captions)))
    print("Number of images" + str(len(image_filename)))

    for idx in range(0, len(image_filename), BATCH_SIZE):
        filenames = image_filename[idx:idx + BATCH_SIZE]
        caption = captions[idx:idx + BATCH_SIZE]
        batch_images, batch_captions = get_batch_data_image_caption(filenames, caption, word_embedding_dict,
                                                                    max_sequence_length=30)
        print("New batch data size")
        print(np.shape(batch_images))
        print(np.shape(batch_captions))
