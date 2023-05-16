import numpy as np
import csv
import datasets
import json
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer 
from transformers import pipeline

nerdict = {'B-CARDINAL':0 , 'B-DIS': 1, 'B-LOC':2, 'B-MED':3, 'B-ORG':4, 'B-ORGANISM':5, 'B-OTH':6, 'B-PER':7, 'B-TIME':8, 'I-CARDINAL':9, 'I-DIS':10, 'I-LOC':11, 'I-MED':12, 'I-ORG':13, 'I-ORGANISM':14, 'I-OTH':15, 'I-PER':16, 'I-TIME':17, 'O':18}
dataset = []

with open("dataset-medical.csv") as f:
  reader = csv.DictReader(f)
  sc = 0  #sentence-count
  tokens = [] #tokens in a sentence
  ner_tags = [] #index of tags

  for row in reader:
    if row['word'] == '.':
      entry = {}
      entry['id'] = str(sc)
      entry['tokens'] = tokens
      entry['ner_tags'] = ner_tags
      dataset.append(entry)
      tokens = []
      ner_tags = []
      sc = sc + 1
    
    else:
      t = row['word']
      if '\u200c' in t:
        t = t.replace('\u200c', '')

      tokens.append(t)
      ner_tags.append(nerdict[row['tag']])

# print(dataset)

train = []
validation = []
test = []
n = len(dataset)

print("Total data size:",n)

train = []
validation = []
test = []
n = n // 2    # dividing dataset into three parts

for i in range(len(dataset)):
  if i < n:
    train.append(dataset[i])
  elif i < n + (n // 2):
    validation.append(dataset[i])
  else:
    test.append(dataset[i])

print("Training data size:",len(train))
print("Validation set size:",len(validation))
print("Testing data size:",len(test))

print(train[0])

d = {'train':Dataset.from_list(train),
     'validation':Dataset.from_list(validation),
     'test':Dataset.from_list(test)
     }

d = DatasetDict(d)
print(d)

nerLabels = ['B-CARDINAL', 'B-DIS', 'B-LOC', 'B-MED', 'B-ORG', 'B-ORGANISM', 'B-OTH', 'B-PER', 'B-TIME', 'I-CARDINAL', 'I-DIS', 'I-LOC', 'I-MED', 'I-ORG', 'I-ORGANISM', 'I-OTH', 'I-PER', 'I-TIME', 'O']

from datasets import Features, Sequence, ClassLabel

d['train'].features['ner_tags'] = Sequence(feature=ClassLabel(num_classes=19, names=nerLabels))
d['test'].features['ner_tags'] = Sequence(feature=ClassLabel(num_classes=19, names=nerLabels))
d['validation'].features['ner_tags'] = Sequence(feature=ClassLabel(num_classes=19, names=nerLabels))

d['train'].features

tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

example_text = d['train'][0]

tokenized_input = tokenizer(example_text["tokens"], is_split_into_words=True)

tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

word_ids = tokenized_input.word_ids()

print(word_ids)

''' As we can see, it returns a list with the same number of elements as our processed input ids, mapping special tokens to None and all other tokens to their respective word. This way, we can align the labels with the processed input ids. '''

print(tokenized_input)

print(len(example_text['ner_tags']), len(tokenized_input["input_ids"]))

def tokenize_and_align_labels(examples, label_all_tokens=True): 
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token                 
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 

q = tokenize_and_align_labels(d['train'][4:5]) 
print(q) 

for token, label in zip(tokenizer.convert_ids_to_tokens(q["input_ids"][0]),q["labels"][0]): 
    print(f"{token:_<40} {label}") 

tokenized_datasets = d.map(tokenize_and_align_labels, batched=True)
model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=19)

args = TrainingArguments( 
"test-ner",
evaluation_strategy = "epoch",
learning_rate=2e-5, 
per_device_train_batch_size=16, 
per_device_eval_batch_size=16, 
num_train_epochs=5, 
weight_decay=0.01, 
) 

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

example = d['train'][0]
print(example)

d['train'].features

labels = [nerLabels[i] for i in example["ner_tags"]] 

metric.compute(predictions=[labels], references=[labels]) 

def compute_metrics(eval_preds): 
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [nerLabels[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [nerLabels[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
   ] 
    results = metric.compute(predictions=predictions, references=true_labels) 
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  } 

trainer = Trainer( 
    model, 
    args, 
   train_dataset=tokenized_datasets["train"], 
   eval_dataset=tokenized_datasets["validation"], 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics 
) 

trainer.train() 

model.save_pretrained("ner_model_for_telugu")
tokenizer.save_pretrained("tokenizer")
id2label = {
    str(i): label for i,label in enumerate(nerLabels)
}

label2id = {
    label: str(i) for i,label in enumerate(nerLabels)
}

config = json.load(open("ner_model_for_telugu/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model_for_telugu/config.json","w"))
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model_for_telugu")
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)


example = "ప్రేమ్ కుమార్ ఆగస్టు మంగళవారం నుండి సుముఖ్‌తో పని చేస్తున్నప్పుడు ముంబైలో ప్రసిద్ధ సంగీత దర్శకుడు."
# example = "అందుకు కాంగ్రెస్తో కలిసి చెట్టాపట్టాల్‌ వేయడానికి కూడా వెనుకాడబోనని అరవింద్‌ కేజ్రీవాల్‌ సంకేతాలు పంపించటం విశేషం"

ner_results = nlp(example)

for wordTag in ner_results:
    print(wordTag)