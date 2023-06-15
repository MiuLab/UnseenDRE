# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

import json
import re

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer


n_class = 1
reverse_order = False
sa_step = False
MAX_LEN = 512


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
map_pair = {
    "per:positive_impression":"person positive impression",
    "per:negative_impression":"person negative impression",
    "per:acquaintance":"person acquaintance",
    "per:alumni":"person alumni",
    "per:boss":"person boss",
    "per:subordinate":"person subordinate",
    "per:client":"person client",
    "per:dates":"person dates",
    "per:friends":"person friends",
    "per:girl/boyfriend":"person girl boyfriend",
    "per:neighbor":"person neighbor",
    "per:roommate":"person roommate",
    "per:children":"person children",
    "per:other_family":"person other family",
    "per:parents":"person parents",
    "per:siblings":"person siblings",
    "per:spouse":"person spouse",
    "per:place_of_residence":"person place of residence",
    "per:place_of_birth":"person place of birth",
    "per:visited_place":"person visited place",
    "per:origin":"person origin",
    "per:employee_or_member_of":"person employee or member of",
    "per:schools_attended":"person schools attended",
    "per:works":"person works",
    "per:age":"person age",
    "per:date_of_birth":"person date of birth",
    "per:major":"person major",
    "per:place_of_work":"person place of work",
    "per:title":"person title",
    "per:alternate_names":"person alternate names",
    "per:pet":"person pet",
    "gpe:residents_of_place":"geopolitics entity residents of place",
    "gpe:births_in_place":"geopolitics entity births in place",
    "gpe:visitors_of_place":"geopolitics entity visitors of place",
    "org:employees_or_members":"organization employees or members",
    "org:students":"organization students",
}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, text_d=None, id_n=None, text_e=None):
        self.guid = guid
        self.text_a = text_a # dialogue
        self.text_b = text_b # x
        self.text_c = text_c # y
        self.text_d = text_d # hint(relation class)
        self.text_e = text_e # trigger word
        self.id_n = id_n
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, id_n, t_idx, x_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.id_n = id_n
        self.t_idx = t_idx
        self.x_idx = x_idx



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class bertProcessor(DataProcessor): #bert
    def __init__(self):
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(3):
            with open("class_3_balance_on_A/"+["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid=data[i][1][j]["rid"][0]
                    # determine hint (try catch for unanswerable)
                    if data[i][1][j]["r"][0].lower()=="unanswerable":
                        hint = data[i][1][j]["r"][0].lower()
                    else:
                        hint = data[i][1][j]["r"][0].lower()
                    hint = map_pair[hint]

                    trigger = data[i][1][j]['t'][0].lower()
                    id_n = data[i][1][j]["id"]
                    d = ['\n'.join(data[i][0]).lower(),
                         data[i][1][j]["x"].lower(),
                         data[i][1][j]["y"].lower(),
                         rid,
                         hint,
                         id_n,
                         trigger]
                    self.D[sid] += [d]
                
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(data[i][0])
            text_b = tokenization.convert_to_unicode(data[i][1])
            text_c = tokenization.convert_to_unicode(data[i][2])
            text_d = tokenization.convert_to_unicode(data[i][4]) # text_d: relation type eg. per:alternate_names
            text_e = tokenization.convert_to_unicode(data[i][6]) # trigger word
            # id_n = tokenization.convert_to_unicode(data[i][5])
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c, text_d=text_d, id_n=data[i][5], text_e=text_e))


            
        return examples


def tokenize(text, tokenizer):
    D = ['[unused1]', '[unused2]']
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in ['[unused1]', '[unused2]']:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):

        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)
        tokens_d = tokenize(example.text_d, tokenizer)
        tokens_e = tokenize(example.text_e, tokenizer)

        # _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4) # 4 means for 3 [SEP] and 1 [CLS]
        # tokens_b = tokens_b + ["[SEP]"] + tokens_c
        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, tokens_d, max_seq_length - 5)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c + ["[SEP]"] + tokens_d

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        # print(len(tokens))
        # exit()

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to
        input_mask = [1] * len(input_ids)

        ######################################################################################################
        # Check d.find > 512
        # check tokenizer(d).input_ids == tokenizer.convert_tokens_to_ids(d) ??
        # trigger data
        t_start = 0
        t_end = 1
        d = example.text_a # only use for d.find(trigger) because truncate 512
        trigger = example.text_e

        if len(trigger) > 0:
            t_start = d.find(trigger)
            t_start = len( tokenizer.convert_tokens_to_ids(tokenize(d[:t_start], tokenizer))  )
            
            trigger_ids = tokenizer.convert_tokens_to_ids(tokenize(trigger, tokenizer))
            d_ids = tokenizer.convert_tokens_to_ids(tokens_a)
            for k in range(len(d_ids)):
                if d_ids[k : k + len(trigger_ids)] == trigger_ids:
                    # t_start = k  # false
                    t_start = k + 1 # true
                    break
            try:
                t_end = t_start + len(tokenizer.tokenize(trigger))
            except:
                ipdb.set_trace()
            # print(d_ids[t_start:t_end])
            # exit()
            # handle trigger location outside 512 max length constraint
            if t_start > len(tokenizer.convert_tokens_to_ids(tokens_a))+1-len(tokenizer.convert_tokens_to_ids(tokenize(trigger, tokenizer))):
                t_start = 0
                t_end = 1
            

        x_st = len(tokenizer.convert_tokens_to_ids(tokens_a)) + 2 # [CLS] and [SEP]
        x_nd = x_st + len(tokenizer.convert_tokens_to_ids(tokenize(example.text_b, tokenizer)))
        ######################################################################################################
        
        ##------------------------------------------------------------------------------------------------
        # print("test trigger word")
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # gt_trigs = tokenizer.decode( input_ids[t_start:t_end] )
        # print("|{}|  |{}|".format(trigger, gt_trigs))
        # print(gt_trigs)
        # exit()



        ##------------------------------------------------------------------------------------------------


        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label 
        id_n = example.id_n
        
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [tokenization.printable_text(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        
        # [['a'], ['b'], []] : [0][0] [1][0] ...
        # [['a], ['b], []] -> [['a], ['b], ['c'], []]
        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        id_n=id_n,
                        t_idx=[t_start,t_end],
                        x_idx=[x_st,x_nd]))
        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]

    return features



def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, tokens_d, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c) + len(tokens_d)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()            




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")

    args = parser.parse_args()
    processors = {"bert": bertProcessor}
    #################################### begin ####################################################################
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        if args.do_train and not args.resume:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    ################################# end #######################################################################




    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / n_class / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, 1)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    if args.fp16:
        model.half()
    model.to(device)
    





    ################################## begin ######################################################################
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    global_step = 0
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    ################################## end ######################################################################




    if args.do_train:
        
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        input_ids = []
        input_mask = []
        segment_ids = []
        label_id = []
        id_n = []

        t_idx = []
        x_idx = []
        # trigger_word = []
        for f in train_features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            t_idx.append([])
            x_idx.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)
                t_idx[-1].append(f[i].t_idx)
                x_idx[-1].append(f[i].x_idx)
                # trigger_word.append(f[i].trigger_word)
            label_id.append([f[0].label_id])      
            id_n.append([f[0].id_n])          

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_id, dtype=torch.float)
        all_id_ns = torch.tensor(id_n, dtype=torch.long)
        all_t_idx = torch.tensor(t_idx, dtype=torch.long)
        all_x_idx = torch.tensor(x_idx, dtype=torch.long)
        # all_trigger_word = torch


        ########################################### begin #############################################################
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_id_ns, all_t_idx, all_x_idx)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        ############################################ end ############################################################

        for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids, id_ns = batch
                input_ids, input_mask, segment_ids, label_ids, id_ns, t_idx, x_idx = batch
                # print(input_ids)
                # print(x_idx)
                # exit()
                # loss, _ = model(input_ids, segment_ids, input_mask, label_ids, 1)
                # loss, _ = model(input_ids, segment_ids, input_mask, label_ids, 1, torch.tensor([2]).to(device))
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, 1, torch.tensor([2]).to(device), t_idx, x_idx)



                ############################################# begin ###########################################################
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale 
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1
                ############################################### end #########################################################
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_{}.pt".format(epoch_num)))
            continue

if __name__ == "__main__":
    main()
