Spam Content Detection and Classification using Naive Bayes Classifier
---------------------------------------------------
## Objective:
The spam content increases as people extensively use social media and the time spent by people using social media is also overgrowing, especially during the time of the pandemic. While users get a lot of text messages through social media they cannot recognize the spam content in these messages. To improve social media security, the detection and control of spam text are essential. Therefore, we want to choose spam content detection as the main topic of this project. Naïve Bayes algorithm will be used for learning and classification of messages as spam and ham. We will use some features to train the models and compare the results.

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

![image](https://github.com/yingliu1206/spam_detection/blob/main/plots/data_description_1.png)
* There are just two labels: ham and spam.
* From all 5572 messages, there are 5169 unique messages.
* The number of ham messages is 4825.
* The most frequent message is "Sorry, I'll call later", which shows 30 times.

We make a bar chart to detect how long the text messages are. Almost all the messages are below 200 characters.
![image](https://github.com/yingliu1206/spam_detection/blob/main/plots/data_description_2.png)


## Model Selection -- Multinomial Naive bayes
Multinomial Naïve Bayes algorithm will be used for learning and classification of messages as spam and ham. 
Why Multinomial Naive Bayes classifier algorithm is a good choice for spam filtering?

* Bayes theorem has strong independence property and it gives the probability of an event based on the prior knowledge of a related event. 

* Multinomial  Naive Bayes model: a document can be represented by a feature vector with integer elements whose value is the frequency of that word in the document.  The document feature vectors capture the frequency of words, not just their presence or absence. 


Finally, accuracy, precision, recall, F1-score, and support cases (how many cases supported that classification) will be used to evaluate the model performance.

## Results


| Model Name | Accuracy | Precision | Recall | F1     |
|------------|----|-----------|-------|--------|
| base line  |    |    |  |  |
| model      |  |   |  |  |





`python pi_logreg.py --sts_dev_file paraphrase-identification-CassieLuo1/stsbenchmark/sts-dev.csv --sts_train_file /paraphrase-identification-CassieLuo1/stsbenchmark/sts-train.csv`

##

`threshold_baseline.py` converts a STS dataset to paraphrase identification


`python threshold_baseline.py --sts_data stsbenchmark/sts-dev.csv --cos_sim_threshold 0.8`


# Discussion
