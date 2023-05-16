import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score


class Dataset(object):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        print(self.df)
        self.sentences = self.get_sentences()

    def get_sentences(self):
        agg = lambda group: [
            (word, tag)
            for word, tag in zip(group['word'].values.tolist(), group['tag'].values.tolist())
        ]
        self.grouped = self.df.groupby("sentence_id").apply(agg)

        return [sentence for sentence in self.grouped]

    def get_features(self):
        features = []
        for sentence in self.sentences:
            sentence_feature = [
                {
                    'bias': 1.0,
                    'word': word,
                }
                for (word, _) in sentence
            ]
            features.append(sentence_feature)

        return features

    def get_labels(self):
        labels = []
        for sentence in self.sentences:
            sentence_labels = [
                tag
                for (_, tag) in sentence
            ]
            labels.append(sentence_labels)

        return labels


def get_trained_model(X_train, y_train):
    """
    Training CRF Classifier
    """
    classifier = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=1000,
        all_possible_transitions=False,
        verbose=True,
    )
    classifier.fit(X_train, y_train)

    return classifier


dataset_domain1 = Dataset("dataset-generic.csv")
dataset_domain2 = Dataset("medical-dataset.csv")
X_train, y_train = (dataset_domain1.get_features(), dataset_domain1.get_labels())
X_test, y_test = (dataset_domain2.get_features(), dataset_domain2.get_labels())
# classes = np.unique(y).tolist()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
classifier = get_trained_model(X_train, y_train)

y_pred = classifier.predict(X_test)

report = metrics.flat_classification_report(y_test, y_pred)
print("================================== SKLEARN CLASSIFICATION REPORT ======================================")
print(report)
print("================================== SEQEVAL CLASSIFICATION REPORT ======================================")
print(classification_report(y_test, y_pred))
print("================================== SEQEVAL F1 SCORE  ======================================")
print(f1_score(y_test, y_pred))
print("================================== SEQEVAL PRECISION  ======================================")
print(precision_score(y_test, y_pred))
print("================================== SEQEVAL RECALL  ======================================")
print(recall_score(y_test, y_pred))
print("================================== SEQEVAL ACCURACY  ======================================")
print(accuracy_score(y_test, y_pred))
