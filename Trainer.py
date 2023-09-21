import random
import time
import torch.optim
from matplotlib import pyplot as plt
from Model import sentiment_model
from datapreparation import Processor
from transformers import BertTokenizer, BertModel, RobertaTokenizer, \
    RobertaModel, DistilBertModel, DistilBertTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import argparse
import transformers
from sklearn.metrics import confusion_matrix
import seaborn as sns

transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser(description="Train a Sentiment classifier - via Transformers")
parser.add_argument("--BERT", type=bool, help="You are using BERT", default=False)
parser.add_argument("--RoBERTa", type=bool, help="You are using RoBERTa", default=False)
parser.add_argument("--DistilBERT", type=bool, help="You are using DistilBERTa", default=True)

parser.add_argument("--epoch", type=int, help="this is the number of epochs", default=10)
parser.add_argument("--hidden_size", type=int, help="this is the LSTM hidden_size", default=100)
parser.add_argument("--batch_size", type=int, help="number of samples in each iteration", default=50)
parser.add_argument("--lr", type=float, help="this is learning rate value", default=0.000001)
parser.add_argument("--max_length", type=int, help="this is maximum length of an utterance", default=100)
parser.add_argument("--n_cluster", type=int, help="number of clusters in kmeans", default=3)

parser.add_argument("--L1_reg", type=bool, help="L1 regularizer", default=False)
parser.add_argument("--L2_reg", type=bool, help="L2 regularizer", default=True)
parser.add_argument("--projection", type=bool, help="apply random projection", default=True)
parser.add_argument("--drop_out", type=bool, help="implement a dropout to the model output", default=True)

parser.add_argument("--L1_lambda", type=int, help="Lambda value used for regularization", default=0.01)
parser.add_argument("--L2_lambda", type=int, help="Lambda value used for regularization", default=0.02)
parser.add_argument("--p", type=int, help="Lambda value used for regularization", default=0.5)
parser.add_argument("--projection_value", type=int, help="change the embedding out dimension", default=100)

args = parser.parse_args()

if args.BERT:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", num_labels=args.n_cluster)
elif args.RoBERTa:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='RoBERT_CacheDir')
    model = RobertaModel.from_pretrained('roberta-base', num_labels=args.n_cluster)
elif args.DistilBERT:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', num_labels=args.n_cluster)

if args.projection:
    model_object = sentiment_model(args.projection_value, args.hidden_size, args.n_cluster, args.p)
else:
    embed_out = model.config.hidden_size
    model_object = sentiment_model(embed_out, args.hidden_size, args.n_cluster, args.p)

if args.L2_reg:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr, weight_decay=args.L2_lambda)
else:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr)

loss = CrossEntropyLoss()
processor_object = Processor(args.n_cluster, model, tokenizer, args.projection_value)

def train_classifier():
    loss_records = []
    patience = 0
    curr_loss = 0

    train_data_batch, train_label_batch = processor_object.main('train.tsv', args.batch_size, args.max_length, args.projection)
    dev_data_batch, dev_label_batch = processor_object.main('dev.tsv', args.batch_size, args.max_length, args.projection)
    test_data_batch, test_label_batch = processor_object.main('test.tsv', args.batch_size, args.max_length, args.projection)

    for epoch_indx in range(args.epoch):
        prev_loss = curr_loss
        epoch_loss = 0
        acc_epoch_record = []
        loss_epoch_record = []
        train_text, train_labels = randomize(train_data_batch, train_label_batch)
        for batch_indx in tqdm(range(len(train_labels)),desc=f"TRAINING DATASET: {epoch_indx + 1}/{args.epoch}"):
            batched_data = train_text[batch_indx]
            gold_label = train_labels[batch_indx]
            gold_label_tensor = torch.from_numpy(gold_label)
            predicted = model_object.forward(batched_data,args.drop_out)
            preds = torch.argmax(predicted, dim=1)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
            acc_epoch_record.append(accuracy)
            predicted = predicted.clone().detach().requires_grad_(True).float()
            gold_label_tensor = gold_label_tensor.clone().detach().long()
            loss_value = loss(predicted, gold_label_tensor)
            if args.L1_reg:
                for param in model.parameters():
                    loss_value += torch.sum(torch.abs(param)) * args.L1_lambda
            loss_epoch_record.append(loss_value.item())
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Ave TRAIN acc: Epoch {epoch_indx + 1}: {sum(acc_epoch_record) / len(acc_epoch_record)}")
        print(f"Ave TRAIN loss: Epoch {epoch_indx + 1}: {sum(loss_epoch_record) / len(loss_epoch_record)}")
        loss_records.append(epoch_loss)
        eval_text, eval_labels = randomize(dev_data_batch, dev_label_batch )
        acc, ave_loss, _,_ = evaluation(model_object, eval_text, eval_labels)
        print(f"Ave DEV acc: Epoch {epoch_indx + 1}: {acc}")
        print(f"Ave DEV loss: Epoch {epoch_indx + 1}: {ave_loss}")
        curr_loss = ave_loss
        if curr_loss >= prev_loss:
            patience += 1
            if patience > 1:
                acc, ave_loss, total_predicted_label,total_gold_label = evaluation(model_object,test_data_batch, test_label_batch)
                print(f"Ave TEST acc: {acc}")
                print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
                matrices(total_gold_label,total_predicted_label)
                return
        else:
            patience = 0
    acc, ave_loss,total_predicted_label,total_gold_label = evaluation(model_object, test_data_batch, test_label_batch)
    print(f"Ave TEST acc: {acc}")
    print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
    matrices(total_gold_label,total_predicted_label)

def evaluation(model,text,label):
    ave_loss_epoch = []
    total_predicted_label = []
    total_gold_label = []
    with torch.no_grad():
        for batch_indx in range(len(text)):
            text_batch = text[batch_indx]
            gold_label_list = label[batch_indx]
            total_gold_label.extend(gold_label_list)
            gold_label_tensor = torch.tensor(gold_label_list)
            predicted = model.forward(text_batch, drop_out=False)
            predicted = predicted.clone().float()
            gold_label_tensor = gold_label_tensor.clone().long()
            loss_value = loss(predicted, gold_label_tensor)
            ave_loss_epoch.append(loss_value)
            preds = torch.argmax(predicted, dim=1)
            total_predicted_label.extend(preds)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
        return accuracy, sum(ave_loss_epoch)/len(ave_loss_epoch), total_predicted_label,total_gold_label

def plotting(records):
    batchList = [i for i in range(len(records))]
    plt.plot(batchList, records, linewidth=5, label="Loss variation")
    plt.xlabel("Batch", color="green", size=20)
    plt.ylabel("Loss", color="green", size=20)
    plt.title("Progress Line for BERT Model", size=20)
    plt.grid()
    plt.show()

def randomize(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return shuffled_list1,shuffled_list2

def calculate_accuracy(gold_label, predicted):
    correct = torch.sum(gold_label== predicted).item()
    total = len(gold_label)
    accuracy = (correct / total) * 100
    return accuracy

def matrices(gold, predicted):
    results = classification_report(gold, predicted)
    print(results)
    cm = confusion_matrix(gold, predicted)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    train_classifier()
    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))
