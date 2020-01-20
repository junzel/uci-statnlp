import pdb

from speech import *

print("Reading data")
tarfname = "data/speech.tar.gz"
# speech = read_files_params(tarfname, ngram_range=(1,3))
speech = read_files_TFIDF(tarfname)
print("Training classifier")
import classify
cls = classify.train_classifier(speech.trainX, speech.trainy)
print("Evaluating")
classify.evaluate(speech.trainX, speech.trainy, cls)
dev_acc = classify.evaluate(speech.devX, speech.devy, cls)

if dev_acc > 0.42:
    print("Reading unlabeled data")
    unlabeled = read_unlabeled(tarfname, speech)
    print("Writing pred file")
    write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)
else:
    pass

# You can't run this since you do not have the true labels
# print "Writing gold file"
# write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
# write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")
