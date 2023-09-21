import torch
import transformers
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
transformers.logging.set_verbosity_error()

projectionModel = SparseRandomProjection(n_components= 200)
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_sentences(data_name):
    data = pd.read_csv(data_name, delimiter='\t')
    text = data.text
    sentences = [x for x in text]
    sentences = sentences[:2000]
    return sentences


def batchify(data, batch_size):
    batched_data = DataLoader(data, batch_size=batch_size)
    return batched_data

def sentence_to_embedding(batched_data, max_length):

    embedding_list = []
    for batch in tqdm(batched_data, desc= f"Creating Embeddings"):
        encoded_sentence = tokenizer.batch_encode_plus(batch, return_tensors='pt', truncation=True, padding='max_length',max_length=max_length)
        inputs = encoded_sentence['input_ids']
        outputs = model(inputs)[0]
        sentence_embeddings= torch.mean(outputs, dim=1)
        sentence_embeddings = sentence_embeddings.clone().detach()
        compressed_outputs = projectionModel.fit_transform(sentence_embeddings)
        sentence_embeddings = torch.from_numpy(compressed_outputs).to(torch.float32)
        embedding_list.append(sentence_embeddings)
    concat_tensor = torch.cat(embedding_list, dim=0)
    return concat_tensor


def test_n_cluster(data, number):
    cluster_range = range(1, number)
    wcss = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(cluster_range, wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squares')
    plt.show()


if __name__ == '__main__':
    datset = get_sentences('dev.tsv')
    batched_data = batchify(datset, 20)
    embeddings = sentence_to_embedding(batched_data, 100)
    test_n_cluster(embeddings, 5)
