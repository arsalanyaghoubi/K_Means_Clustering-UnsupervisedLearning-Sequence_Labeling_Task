import transformers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import seaborn as sns

transformers.logging.set_verbosity_error()
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from sklearn.random_projection import SparseRandomProjection

class Processor():
    def __init__(self, n_clusters, model, tokenizer, projection_size):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        self.n_clusters = n_clusters
        self.model = model
        self.tokenizer = tokenizer
        self.projectionModel = SparseRandomProjection(n_components= projection_size)
        self.gold_labels = []
        self.unsupervised_predicted_label_list = []

    def get_sentences(self, data_name):
        data = pd.read_csv(data_name, delimiter='\t')
        text = data.text
        sentiments = data.sentiment
        sentences = [x for x in text]
        self.gold_labels = [x for x in sentiments]
        return sentences

    def batchify(self, data, batch_size):
        batched_data = DataLoader(data, batch_size=batch_size)
        return batched_data

    def sentence_to_embedding(self, batched_data, max_length, name, batch_size,check_projection):
        labels_list = []
        embedding_list = []
        with open(f"{name}_cached_labels_{self.n_clusters}#C_{batch_size}#B", "w", encoding='utf-8') as cache_file:
            for batch in tqdm(batched_data, desc= f"Creating Embeddings and Labels for {name}"):
                encoded_sentence = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', truncation=True, padding='max_length',max_length=max_length)
                inputs = encoded_sentence['input_ids']  # batch_size * max_length
                outputs = self.model(inputs)[0]  # batch * max_length * embedd_out
                sentence_embeddings = torch.mean(outputs, dim=1)
                sentence_embeddings = sentence_embeddings.clone().detach()
                if check_projection:
                    compressed_outputs = self.projectionModel.fit_transform(sentence_embeddings)
                    sentence_embeddings = torch.from_numpy(compressed_outputs).to(torch.float32)
                embedding_list.append(sentence_embeddings)
                labels = self.k_mean_cluster_function(sentence_embeddings)
                cache_file.write(','.join(str(elem) for elem in labels) + '\n')
                labels_list.append(labels)
        self.unsupervised_predicted_label_list = labels_list
        return labels_list, embedding_list

    def k_mean_cluster_function(self, sentence_tensor):
        cluster_labels = self.kmeans.fit_predict(sentence_tensor)
        return cluster_labels

    def main(self, data_name, batch_size, max_length, check_projection):
        sentences = self.get_sentences(data_name)
        if data_name.endswith('train.tsv'):
            sentences = sentences[:60]
        elif data_name.endswith('dev.tsv'):
            sentences = sentences[:80]
        else:
            sentences = sentences[:20]
        batched_data = self.batchify(sentences, batch_size)
        labels_list, embed_list = self.sentence_to_embedding(batched_data, max_length,data_name, batch_size, check_projection)
        return embed_list, labels_list


# if __name__ == '__main__':
#     bert_model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     output_size = bert_model.config.hidden_size
#     class_object = Processor(n_clusters=3, model=bert_model, tokenizer=tokenizer, projection_size=30)
#     name = 'train.tsv'
#     batch_szie = 10
#     max_length = 50
#     projection = True
#     class_object.main(name,batch_szie,max_length,projection)
