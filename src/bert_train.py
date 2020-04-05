from __future__ import print_function


import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

subprocess.call([sys.executable, "-m", "pip", "install", "gluonnlp==0.9.1"])


import time
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock, nn


import gluonnlp as nlp
from gluonnlp.data import BERTSentenceTransform


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

class BERTClassifier(HybridBlock):
    def __init__(self, bert, num_classes=2, dropout=0.0, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert

        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)
      
class BERTDatasetTransform(object):
    def __init__(self, tokenizer, max_seq_length, class_labels=None,
                     label_alias=None, pad=True, pair=True, has_label=True):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'

        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]

        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
                # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
            return input_ids, valid_length, segment_ids, label
        else:
            return self._bert_xform(line) 
        
def get_train_data(train_dir):
    return np.load(os.path.join(train_dir,'train.npy'))
    
def get_test_data(test_dir):
    return np.load(os.path.join(test_dir,'test.npy'))


def save(net, vocabulary, model_dir):
    # model_dir will be empty except on primary container

    json_vocab=vocabulary.to_json()
    with open(os.path.join(model_dir, 'vocab.json'), "w") as out:
        json.dump(json_vocab, out, indent=4, ensure_ascii=True)
    print('Vocabulary saved to "%s"', model_dir)
    net.export(os.path.join(model_dir, 'bert'))

    
def evaluate(model, dataloader, ctx, loss_function):
    metric = mx.metric.Accuracy()
    step_loss = 0
    
    ctx_0 = mx.gpu()
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
        token_ids = token_ids.as_in_context(ctx_0)
        valid_length = valid_length.as_in_context(ctx_0)
        segment_ids = segment_ids.as_in_context(ctx_0)
        label = label.as_in_context(ctx_0)

        out = model(token_ids, segment_ids, valid_length.astype('float32'))
        ls = loss_function(out, label).mean()

        step_loss += ls.asscalar()
        metric.update([label], [out])

    return metric.get()[1], step_loss / len(dataloader)

def train(model, ctx, args, train_dataloader, test_dataloader, loss_function, trainer):
    metric = mx.metric.Accuracy()
    # Collect all differentiable parameters for gradient clipping
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']

    t0 = time.time()
    for epoch_id in range(args.epochs):
        metric.reset()
        step_loss = 0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        
            token_ids = mx.gluon.utils.split_and_load(token_ids, ctx_list=ctx, even_split=False)
            valid_lengths = mx.gluon.utils.split_and_load(valid_length, ctx_list=ctx, even_split=False)
            segment_ids = mx.gluon.utils.split_and_load(segment_ids, ctx_list=ctx, even_split=False)
            labels = mx.gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)

            with mx.autograd.record():
                outs = [model(token_ids_slice, segment_ids_slice, valid_length_slice.astype('float32')) 
                        for (token_ids_slice, segment_ids_slice, valid_length_slice) in zip(token_ids, segment_ids, valid_lengths)]
                lss = [loss_function(out_slice, label_slice).mean() for (out_slice, label_slice) in zip(outs, labels)]
                
            for loss in lss:
                loss.backward()                                                                               

            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, args.grad_clip)
            trainer.update(1)

            step_loss += loss.asscalar()
            
            for (label_slice, out_slice) in zip(labels, outs):
                metric.update([label_slice], [out_slice])

            if (batch_id + 1) % (args.log_interval) == 0:
                tt = time.time()- t0
                print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f} in {} seconds'
                             .format(epoch_id, batch_id + 1, len(train_dataloader),
                                     step_loss / args.log_interval, trainer.learning_rate, metric.get()[1], tt))
                step_loss = 0
                t0 = time.time()
#                 break
        t0 = time.time()
        test_acc, test_loss = evaluate(model, test_dataloader, ctx, loss_function)
        t1 = time.time() - t0
        print('[Epoch {}] test_loss={:.4f}, test_acc={:.3f} in {}'
             .format(epoch_id, test_loss, test_acc, t1))
             
    return model

def main(model, vocabulary, ctx, train_dataset, test_dataset, args):

    # Use the vocabulary from pre-trained model for tokenization
    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

    # The maximum length of an input sequence
    max_len = 500

    # The labels for the two classes
    all_labels = [0, 1]

    transform = BERTDatasetTransform(bert_tokenizer, max_len,
                                     class_labels=all_labels,
                                     has_label=True,
                                     pad=True,
                                     pair=False)

    data_train = train_dataset.transform(transform)
    data_test = test_dataset.transform(transform)

    # Collect all parameters from net and its children, then initialize them.
    trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr, 'epsilon': 1e-9}, update_on_kvstore=False)
    loss_function = mx.gluon.loss.SoftmaxCELoss()

    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=20)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False, num_workers=20)
    
    net=train(model, ctx, args, train_dataloader, test_dataloader, loss_function, trainer)
    #net.save_parameters(os.path.join(args.model_dir, 'bert.params'))
    
    save(net, vocabulary, args.model_dir)



def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT network.")
    
    # Network parameters:

    parser.add_argument("--network", type=str, default="bert_24_1024_16",help="Base Bert network.")
    parser.add_argument("--pretrained", type=bool,default=True, help="Use pretrained weights")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes in training data set.")
    
    parser.add_argument("--batch-size", type=int, default=4, help="Training mini-batch size")
    parser.add_argument("--batch-size-test", type=int, default=4, help="Training mini-batch size")


    # Training process parameters:
    parser.add_argument("--epochs", type=int, default=1,help="The maximum number of passes over the training data.")
    parser.add_argument("--lr", type=float,default=0.0001, help="Learning rate")
    parser.add_argument("--grad-clip", type=float, default=1, help="Grad clip")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--wd", type=float, default=0.0005, help="Weight decay")


    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=int(os.environ["SM_NUM_GPUS"]),help="Number of GPUs to use in training.")
    parser.add_argument("--num-workers", type=int, default=int(os.environ["SM_NUM_CPUS"]),
        help='Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful'
    )
    
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    # I/O Settings:


    parser.add_argument("--log-interval", type=int, default=100,help="Logging mini-batch interval. Default is 100.")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    ##### training contexts ######
    ctx = [mx.gpu(int(i)) for i in range(args.num_gpus)]
    ctx = ctx if ctx else [mx.cpu()]

    #### Define Bert version network #####    
    bert_base, vocabulary = nlp.model.get_model(args.network,
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=args.pretrained, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)

    model = BERTClassifier(bert_base, num_classes=args.num_classes, dropout=0.1)
    
    # only need to initialize the classifier layer.
    model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.hybridize()
    
    
    ###### Get Train and Test data #####

    train_data = get_train_data(args.train)
    test_data = get_test_data(args.test)
    
    def process_label(x):
        data, label = x
        # Label is a review score from 1 to 10. We take 6..10 as a positive sentiment
        # and 1..5 as a negative
        label = int(int(label) > 5)
        return [str(data), label]

    def process_dataset(dataset):
        start = time.time()
        with mp.Pool() as pool:
            # Each sample is processed in an asynchronous manner.
            dataset = gluon.data.SimpleDataset(pool.map(process_label, dataset))
        end = time.time()
        print('Done! Label processing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
        return dataset

    train_data = process_dataset(train_data)
    test_data = process_dataset(test_data)
    
 

    #### training
    main(model, vocabulary, ctx, train_data, test_data, args)


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """   
    model = mx.gluon.SymbolBlock.imports(
        os.path.join(model_dir, 'bert-symbol.json'),
        ["data0","data1","data2"],
        os.path.join(model_dir, 'bert-0000.params')
    )
    
    with open(os.path.join(model_dir, 'vocab.json')) as inp:
        vocabulary = json.load(inp)
    
    logger.info(f"hello model_dir")
    return model, vocabulary


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    logger.info(f"hello transform_fn")    
    net, vocab = net
    
    logger.info(f"got net")
    vocabulary= nlp.vocab.BERTVocab().from_json(vocab)
    logger.info(f"got vocab")
    # Use the vocabulary from pre-trained model for tokenization
    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
    logger.info(f"got tokenizer")
    # The maximum length of an input sequence
    max_len = 500

    # The labels for the two classes
    all_labels = [0, 1]

    transform = BERTDatasetTransform(bert_tokenizer, max_len,
                                     class_labels=all_labels,
                                     has_label=True,
                                     pad=True,
                                     pair=False)
    logger.info(f"got transforme")
    review_text = json.loads(data)
    logger.info(f"got data")
    review_transformed = transform((review_text[0], 0))
    logger.info(f"got text")
    token_ids =  mx.nd.array(review_transformed[0]).reshape(1, -1)
    logger.info(f"got tokens")
    segment_ids =  mx.nd.array(review_transformed[2]).reshape(1, -1)
    valid_length = mx.nd.array(review_transformed[1]).reshape(1)
    
    logger.info(f"got others")
    positive_review_probability = net(token_ids, segment_ids, valid_length.astype('float32')).softmax()  
    logger.info(f"got results")
    response_body = json.dumps({"score":positive_review_probability.asnumpy().tolist()[0]})
    return response_body, output_content_type