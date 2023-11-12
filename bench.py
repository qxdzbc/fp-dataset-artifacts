from dataclasses import dataclass
from typing import Union

import torch
import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from textattack import Attacker
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

# contains the SNLI data set
dataset:Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = None
modelPath = "./trained_model_base_backup"
model: AutoModelForSequenceClassification = None
tokenizer: AutoTokenizer = None
resultList = []

def useTextAttack():
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.attack_recipes import BERTAttackLi2020

    wrappedModel = HuggingFaceModelWrapper(model,tokenizer)
    # attackDataset = HuggingFaceDataset("snli",split="test",)
    originalDataset = datasets.load_dataset("snli",split="test[0%:1%]")
    hansDataSet = datasets.load_dataset("hans",split=datasets.ReadInstruction("train",from_=0, to=5,unit="abs"))
    # attack from item 0->20
    originalDataset = datasets.load_dataset("snli",split=datasets.ReadInstruction("test",from_=0, to=5,unit="abs"))

    attackDataset = HuggingFaceDataset(originalDataset)
    attackDataset = HuggingFaceDataset(hansDataSet)

    # attackDataset = HuggingFaceDataset("snli",split="test")
    ## TODO choose an appropriate attack
    attack = BERTAttackLi2020.build(wrappedModel)
    ## TODO collect statistic of the attack

    print(len(attackDataset))
    attacker = Attacker(attack,attackDataset)
    attackResults = attacker.attack_dataset()
    for rs in attackResults:
        print(rs.__str__(color_method='ansi'))
        # print(json.dumps(rs))

    ## TODO intepretete the attack

    ## TODO make a fix base on the attack

    ## TODO: how to run re-training on CUDA


def loadDataAndModel():
    # load the data
    global dataset
    global model
    global tokenizer
    global resultList
    if dataset is None:
        dataset = datasets.load_dataset("snli")
    # load the model
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(modelPath, **{'num_labels': 3})
    if tokenizer is None:
        # tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        tokenizer = AutoTokenizer.from_pretrained(modelPath, use_fast=True)
    # if len(resultList)==0:
    resultList = loadEvalResult()


class Example:
    def __init__(self, hypothesis, premise, label):
        self.hypothesis = hypothesis
        self.premise = premise
        self.label = label

    def toJson(self):
        return {
            "hypothesis": self.hypothesis,
            "premise": self.premise,
            "label": self.label
        }

    @staticmethod
    def fromJsonObj(jsonObj):
        return Example(
            jsonObj["hypothesis"], jsonObj["premise"], jsonObj["label"]
        )

class PredictionSample:
    def __init__(self, example, predictionLabel):
        self.example = example
        self.predictionLabel = predictionLabel

    def toJson(self):
        return {
            "example": self.example.toJson(),
            "predictedLabel": self.predictionLabel
        }

class WrongContainer:
    def __init__(self):
        self.wrongs = []

    def add(self, wrong):
        self.wrongs.append(wrong)

    def toJson(self):
        return {
            "wrongs": [w.toJson() for w in self.wrongs]
        }


class ExampleContainer:
    def __init__(self, examples):
        self.examples = examples

    def add(self, example: Example):
        self.examples.append(example)

    def toJson(self):
        return {
            "examples": [e.toJson() for e in self.examples]
        }

    @staticmethod
    def fromJsonObj(jsonObj):
        examples = [Example.fromJsonObj(e) for e in jsonObj["examples"]]
        return ExampleContainer(examples)


def exportWrongs():
    wc = WrongContainer()
    for example in dataset["train"]:
        input = tokenizer(example["hypothesis"], return_tensors="pt")
        prediction = model(**input)
        predictedLabel = torch.argmax(torch.softmax(prediction.logits, dim=1)).item()
        if predictedLabel != example["label"]:
            wc.add(PredictionSample(Example(
                example["hypothesis"], example["premise"], example["label"]
            ), predictedLabel))

    with open("resultTrain.json", "w") as file:
        import json
        text = json.dumps(wc.toJson())
        file.write(text)
    print(wc)


def exportRight():
    wc = WrongContainer()
    count = 0
    for example in dataset["train"]:
        if count < 1000:
            input = tokenizer(example["hypothesis"], return_tensors="pt")
            prediction = model(**input)
            predictedLabel = torch.argmax(torch.softmax(prediction.logits, dim=1)).item()
            if predictedLabel == example["label"]:
                wc.add(PredictionSample(Example(
                    example["hypothesis"], example["premise"], example["label"]
                ), predictedLabel))
        else:
            break

    with open("resultTrain_right.json", "w") as file:
        import json
        text = json.dumps(wc.toJson())
        file.write(text)
    print(wc)

def dumpTestData():
    # Dump all test data from the training data set into a file
    data = dataset["test"]
    examples = [Example(example["hypothesis"], example["premise"], example["label"]) for example in data]
    c = ExampleContainer(examples)
    print(len(examples))
    return c
def loadEvalResult():
    filePath = "./data/eval_predictions.jsonl"
    resultList = []
    with open(filePath,"r") as file:
        for line in file.readlines():
            o = json.loads(line)
            resultList.append(o)
    return resultList

def computeCorrectStatistic():
    def resultIsTargetLabel(result,targetLabel):
        label = result["label"]
        predicted = result["predicted_label"]
        return label == targetLabel and label == predicted
    def resultIsEntail(result):
        return resultIsTargetLabel(result,0)
    def resultIsContra(result):
        return resultIsTargetLabel(result,1)
    def resultIsNeutral(result):
        return resultIsTargetLabel(result, 2)

    trueEntails = list(filter(resultIsEntail,resultList))
    trueContra = list(filter(resultIsContra,resultList))
    trueNeutral = list(filter(resultIsNeutral,resultList))
    return (trueEntails,trueContra,trueNeutral)

def computeIncorrectStat():
    def resultIsTargetLabel(result,targetLabel):
        label = result["label"]
        predicted = result["predicted_label"]
        return label == targetLabel and label != predicted
    def resultIsEntail(result):
        return resultIsTargetLabel(result,0)
    def resultIsContra(result):
        return resultIsTargetLabel(result,1)
    def resultIsNeutral(result):
        return resultIsTargetLabel(result, 2)

    entail = list(filter(resultIsEntail,resultList))
    contra = list(filter(resultIsContra,resultList))
    neutral = list(filter(resultIsNeutral,resultList))
    return (entail,contra,neutral)
def loadContrastTest(label):
    contrastFilePath = "/Users/phong/gits/nlp/fp-dataset-artifacts/data/test_data/testData.json"
    with open(contrastFilePath,"r") as file:
        content = file.read()
        contentObj = json.loads(content)
        examples = contentObj["examples"]
        example0 = [
            eg for eg in examples if eg["label"]==label
        ]
        return example0


def findWrongEntailInTestData():
    with open("/Users/phong/gits/nlp/fp-dataset-artifacts/data/wrong_prediction.json","r") as file:
        content = file.read()
        contentObj = json.loads(content)
        for sample in contentObj["wrongs"]:
            if sample["predictedLabel"] == 0:
                example = sample["example"]
                hypo = example["hypothesis"]
                premise = example["premise"]
                if premise.endswith(hypo):
                    print(sample)

def findWrongEntailInTestData():
    with open("/Users/phong/gits/nlp/fp-dataset-artifacts/data/wrong_prediction.json","r") as file:
        content = file.read()
        contentObj = json.loads(content)
        for sample in contentObj["wrongs"]:
            if sample["predictedLabel"] == 0:
                example = sample["example"]
                hypo = example["hypothesis"]
                premise = example["premise"]
                # if premise.endswith(hypo):
                if hypo in premise:
                    print(sample)

@dataclass
class EntailmentBiasInTrainData:
    subsequenceCount = 0
    constituentCount = 0
    lexicalOverlapCount = 0
    totalCount = 0

    def __str__(self):
        return json.dumps({
            "subsequenceCount": self.subsequenceCount,
        "constituentCount":self.constituentCount,
        "lexicalOverlapCount":self.lexicalOverlapCount,
        "totalCount":self.totalCount,
        })

def findBiasDataInTrain(label=0):
    rtLabel0 = EntailmentBiasInTrainData()
    rtLabel1 = EntailmentBiasInTrainData()
    rtLabel2 = EntailmentBiasInTrainData()
    for example in dataset["train"]:
        hypo = example["hypothesis"]
        premise = example["premise"]
        if example["label"] == 0:
            if isSubsequence(hypo, premise):
                rtLabel0.subsequenceCount+=1
            elif isConstituent(hypo, premise):
                rtLabel0.constituentCount+=1
            elif isLexicalOverlap(hypo,premise):
                rtLabel0.lexicalOverlapCount+=1
            rtLabel0.totalCount+=1
        if example["label"] == 1:
            if isSubsequence(hypo, premise):
                rtLabel1.subsequenceCount+=1
            elif isConstituent(hypo, premise):
                rtLabel1.constituentCount+=1
            elif isLexicalOverlap(hypo,premise):
                rtLabel1.lexicalOverlapCount+=1
            rtLabel1.totalCount+=1
        if example["label"] == 2:
            if isSubsequence(hypo, premise):
                rtLabel2.subsequenceCount+=1
            elif isConstituent(hypo, premise):
                rtLabel2.constituentCount+=1
            elif isLexicalOverlap(hypo,premise):
                rtLabel2.lexicalOverlapCount+=1
            rtLabel2.totalCount+=1

    return rtLabel0,rtLabel1,rtLabel2

def findBiasDataInTest():
     with open("/Users/phong/gits/nlp/fp-dataset-artifacts/data/test_data/testData.json","r") as file:
         content = file.read()
         contentObj = json.loads(content)
         for example in contentObj["examples"]:
             hypo = example["hypothesis"]
             premise = example["premise"]
             # if premise.endswith(hypo):
             if hypo in premise:
                 print(example)


def isSubsequence(hypo, premise):
    return premise.endswith(hypo)

def isConstituent(hypo, premise):
    return hypo in premise

def isLexicalOverlap(hypo,premise):
    """
    compute lexical overlap using jaccard similarity
    :param hypo:
    :param premise:
    :return:
    """
    from nltk.tokenize import word_tokenize
    text1 = hypo
    text2 = premise
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    tokens1 = [token.lower() for token in tokens1 if token.isalpha()]
    tokens2 = [token.lower() for token in tokens2 if token.isalpha()]
    overlap = set(tokens1).intersection(set(tokens2))
    jaccard_similarity = float(len(overlap)) / (len(set(tokens1)) + len(set(tokens2)) - len(overlap))
    rt= jaccard_similarity >0.5
    return rt


def evaluateBaseLineModelOnSubSetOfTestData():
    """
    give a breakdown of data in test set (overlap, subsequence, constituent)
    count correct and wrong prediction on test dataset, and also break them down into the 3 categories
    :return:
    """

    overlapDataset = dataset.filter(lambda sample: isLexicalOverlap(sample["hypothesis"],sample["premise"]))
    subsequenceDataset = dataset.filter(lambda sample: isSubsequence(sample["hypothesis"],sample["premise"]))
    constituentDataset = dataset.filter(lambda sample: isConstituent(sample["hypothesis"],sample["premise"]))
    targetDataset = overlapDataset

    correctCount = 0
    wrongCount = 0

    entail = 0
    contrast = 0
    neutral = 0

    entailRight = 0
    contrastRight = 0
    neutralRight = 0

    entailWrong = 0
    contrastWrong = 0
    neutralWrong = 0


    for example in targetDataset["test"]:
        input = tokenizer(example["hypothesis"], return_tensors="pt")
        prediction = model(**input)
        predictedLabel = torch.argmax(torch.softmax(prediction.logits, dim=1)).item()
        label = example["label"]
        if label == 0:
            entail+=1
        elif label == 1:
            contrast +=1
        elif label == 2:
            neutral+=1


        if predictedLabel == label:
            correctCount+=1
            if label == 0:
                entailRight += 1
            elif label == 1:
                contrastRight += 1
            elif label == 2:
                neutralRight += 1
        else:
            wrongCount+=1
            if label == 0:
                entailWrong += 1
            elif label == 1:
                contrastWrong += 1
            elif label == 2:
                neutralWrong += 1
    print(f"""
Base composition:
    - entail: {entail}
    - contrast: {contrast}
    - neutral: {neutral}
        """)

    print(f"""
correct: {correctCount}
    - entail: {entailRight}
    - contrast: {contrastRight}
    - neutral: {neutralRight}
wrong: {wrongCount}
    - entail: {entailWrong}
    - contrast: {contrastWrong}
    - neutral: {neutralWrong}
    """)


def sampleTextAttack():
    import textattack
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    hugDataset = datasets.load_dataset("snli", split=datasets.ReadInstruction("train", from_=0, to=5, unit="abs"))

    dataset = HuggingFaceDataset(hugDataset)
    attack = TextFoolerJin2019.build(model_wrapper)

    rs = Attacker(attack,dataset).attack_dataset()

    for r in rs:
        print(r.__str__(color_method='ansi'))



loadDataAndModel()
# findWrongEntailInTestData()
# useTextAttack()
# findBiasDataInTest()
# entailmentBiasInTrainDataResult_label0,entailmentBiasInTrainDataResult_label1,entailmentBiasInTrainDataResult_label2 = findBiasDataInTrain()
# evaluateBaseLineModelOnSubSetOfTestData()
# evaluateBaseLineModelOnSubSetOfTestData()
sampleTextAttack()