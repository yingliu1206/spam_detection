Spam Content Detection and Classification using Naive Bayes Classifier
---------------------------------------------------
## Objective:
The spam content increases as people extensively use social media and the time spent by people using social media is also overgrowing, especially in the time of the pandemic. While users get a lot of text messages through social media they cannot recognize the spam content in these messages. To improve social media security, the detection and control of spam text are essential. Therefore, we want to choose spam content detection as the main topic in this project.

## Dataset:
* Origin: For the dataset, [MS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) will be used. It is a public set of SMS labeled messages that have been collected for mobile phone spam research. 
* Describe the dataset: 
  * It consists of 5574 instances. 
    * A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. 
    * A subset of 3375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC). 
    * A list of 450 SMS ham messages collected from Caroline Tag’s PhD Thesis. 
    * Finally, 1,002 SMS ham messages and 322 spam messages were collected from the SMS Spam Corpus v.0.1 Big. 
* For attribute information, the dataset is composed of just one text file, where each line has the correct class followed by the raw message.

## Exploratory Data Analysis

![Alt text](https://github.com/yingliu1206/spam_detection/plots/data_description_1.png)


## Results

* Describe the dataset 

  * The training dataset has 4476 rows and 3 columns. 
  * The validation dataset has 1221 rows and 3 columns. 
  * The columns describe three features of each pair of sentences: BLEU, Word Error Rate and Cosine Similarity of TFIDF vectors.
* Describe the baseline (including what threshold you used) and the logistic regression (2+ sentences each), 
  * The baseline: Compute cosine similarity for each pair of sentences, and use 
a threshold of 0.7 to convert each similarity score into a paraphrase prediction. 
Then compare the results with the true labels we got from pi and evaluate the results 
using accuracy, precision, recall and F-1 score. 
  * The logistic regression: Learn a decision boundary on feature representations of input texts 
to best tell classes apart.The logistic regression learns the three features: 
BLEU, Word Error Rate and Cosine Similarity of TFIDF vectors of each pair of sentences and 
their pi labels, and thus can decide on "paraphrase" or "not paraphrase" with new data.
* fill the table with evaluation on the dev partition,
* and compare the results (3 sentences).
  * The logistic regression model has higher accuracy and precision than the baseline model. 
  * The baseline model has higher recall. 
  * The F1 scores of these two models are very similar. 
  * In summary, the logistic regression model performs better than the baseline model.


| Model Name | Accuracy | Precision | Recall | F1     |
|------------|----------|-----------|--------|--------|
| base line  | 0.8321   | 0.6347    | 0.5265 | 0.5756 |
| model      | 0.8206   | 0.5889    | 0.5644 | 0.5764 |

## Homework: pi_logreg.py

* Train a logistic regression for paraphrase identification on the training data using three features:
    - BLEU
    - Word Error Rate
    - Cosine Similarity of TFIDF vectors
* Use the logistic regression implementation in `sklearn`.
* Update the readme as described in *Results*.

`python pi_logreg.py --sts_dev_file paraphrase-identification-CassieLuo1/stsbenchmark/sts-dev.csv --sts_train_file /paraphrase-identification-CassieLuo1/stsbenchmark/sts-train.csv`

## Lab: threshold_baseline.py

`threshold_baseline.py` converts a STS dataset to paraphrase identification
 and checks the distribution of paraphrase/nonparaphrase.
Then, it evaluates TFIDF vector similarity as a model of paraphrase by setting a threshold and
considering all sentence pairs above that that similarity to be paraphrase.

Example usage:

`python threshold_baseline.py --sts_data stsbenchmark/sts-dev.csv --cos_sim_threshold 0.8`


