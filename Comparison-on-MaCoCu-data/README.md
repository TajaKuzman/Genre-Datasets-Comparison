## Comparison of predictions based on domain (URL) information from a test corpus

To analyse how appropriate are various schemata for our target data, which is a MaCoCu web corpus, we apply all the classifiers to a sample of a corpus. Then we analyse the distribution of labels inside one domain. We base this comparison on a hypothesis that web domains mostly consist of only one genre. Thus, we will analyse to which extent the genre labels from different schemata are able to consistently assign a genre to texts inside one domain.

### Data preparation

- We first need to discard all texts with text length smaller than 75 - I created a dictionary of all domains and urls of texts that are long enough.
- Then I calculated the frequency of the domains (number of texts in each domain). I discarded domains that have less than 10 instances (if I wouldn't, the median would be 6 texts per domain). Then I calculated the median and took the instances with the median number of instances, and the same amount of domains above and below the median, so that at the end, the sample has around 1000 different domains. - There were 219 domains at the median, so I took 391 domains from above and below the median.
- Out of the table of domains and all urls belonging to them, I sampled 10 URLs per domain, and extracted texts with these URLs from the MaCoCu-sl XLM file. It turned out that ssome URLs appear multiple times with different texts, so at the end, the sample consisted out of 10.041 texts. The problem with this is a) that some domains have more instances than other, and b) that texts under some of the URLs might be shorter than 75 words. That is why I calculated the length of the texts again and discarded those with length less than 75 words. Then I also sampled out the instances from domains with more than 10 texts, so that at the end all domains have 10 instances.
- The final number of domains is 1001 and number of texts 10,010 (10 per domain). Final file: *MaCoCu-sl-sample.csv*


### Models used

Then I applied the following classifiers, developed in previous experiments and saved to Wandb to the sample:
- FTD classifier - original FTD data (except multi-labeled texts and non-text texts) - 10 categories, 849 instances in training data
- GINCO-downcast classifier - used primary_level_4 downcasted GINCO labels - 9 labels. It was trained on 601 texts.
- CORE-main classifier - main categories only - 9 instances. All texts with multiple labels were discarded. It was trained on 10256 instances.
- GINCO X-GENRE classifier - 9 X-GENRE labels. It was trained on 535 texts (10% texts discarded - belonging to "discarded" labels)
- FTD X-GENRE classifier - 7 X-GENRE labels. It was trained on 630 texts (23% texts were discarded).
- CORE X-GENRE classifier - 9 X-GENRE labels. It was trained on 607 texts - large changes to the dataset were performed (change of distribution, taking only a sample to have a similar size as FTD and GINCO).
- X-GENRE classifier - 9 X-GENRE labels. Trained on the training splits of all of the X-GENRE datasets mentioned above: 1772 instances in the training dataset.

File with the predictions: *MaCoCu-sl-sample-with-predictions.csv*

### Results

#### Comparison of confidence of the predictions

| classifier    |   min |   median |   max |
|-------------|------:|---------:|------:|
| X-GENRE       |  0.29 |     1    |  1    |
| GINCO-X-GENRE |  0.27 |     0.99 |  0.99 |
| GINCO         |  0.18 |     0.94 |  0.98 |
| FTD-X-GENRE   |  0.19 |     0.87 |  0.97 |
| CORE          |  0.23 |     0.86 |  0.99 |
| FTD           |  0.15 |     0.81 |  0.97 |
| CORE-X-GENRE  |  0.15 |     0.53 |  0.95 |

#### Most frequently predicted label

| classifier    | most frequent label                   |   frequency |
|-------------|-------------------------------------|------------:|
| FTD           | A12 (promotion)                       |        0.62 |
| GINCO         | Promotion                             |        0.43 |
| CORE          | Informational Description/Explanation |        0.67 |
| GINCO-X-GENRE | Promotion                             |        0.48 |
| FTD-X-GENRE   | Promotion                             |        0.65 |
| CORE-X-GENRE  | Information/Explanation               |        0.44 |
| X-GENRE       | Promotion                             |        0.43 |

#### Comparison of label distribution (instance level)

| FTD          | GINCO                | CORE                            | GINCO-X-GENRE  | FTD-X-GENRE       | CORE-X-GENRE      | X-GENRE           | |
|------------------------------------|--------------------------------------------|-------------------------------------------------------|--------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|----|
| ('A12 (promotion)', 0.62)       | ('Promotion', 0.43)                     | ('Informational Description/Explanation', 0.67) | ('Promotion', 0.48)               | ('Promotion', 0.65)                  | ('Information/Explanation', 0.44) | ('Promotion', 0.43)                  | |
| ('A16 (information)', 0.12)     | ('Information/Explanation', 0.14)       | ('Informational Persuasion', 0.12)                 | ('Information/Explanation', 0.15) | ('Information/Explanation', 0.16) | ('Instruction', 0.2)                 | ('Information/Explanation', 0.18) |    |
| ('A1 (argumentative)', 0.06) | ('Opinion/Argumentation', 0.11)         | ('Narrative', 0.1)                                 | ('News', 0.13)                    | ('News', 0.06)                       | ('Opinion/Argumentation', 0.14)      | ('News', 0.13)                       | |
| ('A17 (review)', 0.05)          | ('News/Reporting', 0.11)                | ('How-To/Instructional', 0.05)                     | ('Opinion/Argumentation', 0.09)   | ('Instruction', 0.06)                | ('News', 0.13)                       | ('Opinion/Argumentation', 0.11)      | |
| ('A7 (instruction)', 0.05)      | ('List of Summaries/Excerpts', 0.08) | ('Opinion', 0.04)                                  | ('Instruction', 0.07)             | ('Opinion/Argumentation', 0.03)      | ('Forum', 0.06)                      | ('Instruction', 0.08)                | |
| ('A8 (news)', 0.04)             | ('Instruction', 0.07)                   | ('Interactive Discussion', 0.01)                   | ('Other', 0.06)                   | ('Legal', 0.03)                      | ('Other', 0.02)                      | ('Other', 0.03)                      | |
| ('A11 (personal)', 0.03)        | ('Other', 0.03)                         | ('Spoken', 0.01)                                   | ('Forum', 0.01)                   | ('Prose/Lyrical', 0.01)              | ('Prose/Lyrical', 0.02)              | ('Legal', 0.02)                      | |
| ('A9 (legal)', 0.02)            | ('Forum', 0.01)                         | ('Lyrical', 0.0)                                   | ('Legal', 0.01)                   |                                      |                                         | ('Forum', 0.01)                      | |
| ('A4 (fiction)', 0.01)          | ('Legal/Regulation', 0.01)              |                                                    | ('Prose/Lyrical', 0.0)            |                                      |                                         | ('Prose/Lyrical', 0.0)               | |
| ('A14 (academic)', 0.0)         |                                         |                                                       |                                      |                                         |                                         |                                         |    |

#### Comparison of frequency of prediction of the most frequent label per domain

![](figures/Comparison-of-distribution-in-domains-MaCoCu-sl-histogram.png)

![](figures/Comparison-of-distribution-in-domains-MaCoCu-sl-subplots.png)

![](figures/Comparison-of-distribution-in-domains-MaCoCu-sl-KDE.png)


#### Comparison of label distribution on the domain level

Table shows in how many of the domains a label is the most frequent label in the domain. The values in the table are percentages.

|  most frequent label in domain: FTD |  most frequent label in domain: GINCO |  most frequent label in domain: CORE                |  most frequent label in domain: GINCO-X-GENRE |  most frequent label in domain: FTD-X-GENRE |  most frequent label in domain: CORE-X-GENRE |  most frequent label in domain: X-GENRE |
|---------------------------------------|-----------------------------------------|-------------------------------------------------------|-------------------------------------------------|-----------------------------------------------|------------------------------------------------|-------------------------------------------|
|  ('A12 (promotion)', 0.74)          |  ('Promotion', 0.52)                  |  ('Informational Description/Explanation', 0.76)  |  ('Promotion', 0.57)                          |  ('Promotion', 0.76)                        |  ('Information/Explanation', 0.51)           |  ('Promotion', 0.5)                     |
|  ('A16 (information)', 0.08)        |  ('Information/Explanation', 0.13)    |  ('Informational Persuasion', 0.09)                 |  ('News', 0.13)                               |  ('Information/Explanation', 0.12)          |  ('Instruction', 0.18)                       |  ('Information/Explanation', 0.16)      |
|  ('A17 (review)', 0.04)             |  ('News/Reporting', 0.12)             |  ('Narrative', 0.08)                                |  ('Information/Explanation', 0.12)            |  ('News', 0.05)                             |  ('Opinion/Argumentation', 0.13)             |  ('News', 0.14)                         |
|  ('A1 (argumentative)', 0.04)       |  ('Opinion/Argumentation', 0.11)      |  ('How-To/Instructional', 0.03)                     |  ('Opinion/Argumentation', 0.09)              |  ('Instruction', 0.03)                      |  ('News', 0.11)                              |  ('Opinion/Argumentation', 0.11)        |
|  ('A8 (news)', 0.03)                |  ('List of Summaries/Excerpts', 0.04) |  ('Opinion', 0.02)                                  |  ('Instruction', 0.04)                        |  ('Opinion/Argumentation', 0.02)            |  ('Forum', 0.03)                             |  ('Instruction', 0.06)                  |
|  ('A7 (instruction)', 0.03)         |  ('Instruction', 0.04)                |  ('Interactive Discussion', 0.01)                   |  ('Other', 0.03)                              |  ('Legal', 0.01)                            |  ('Prose/Lyrical', 0.02)                     |  ('Other', 0.01)                        |
|  ('A11 (personal)', 0.03)           |  ('Forum', 0.01)                      |  ('Spoken', 0.0)                                    |  ('Forum', 0.01)                              |  ('Prose/Lyrical', 0.01)                    |  ('Other', 0.01)                             |  ('Forum', 0.01)                        |
|  ('A9 (legal)', 0.01)               |  ('Legal/Regulation', 0.01)           |  ('Lyrical', 0.0)                                   |  ('Legal', 0.01)                              |                                             |                                                |  ('Legal', 0.01)                        |
|  ('A4 (fiction)', 0.01)             |  ('Other', 0.0)                       |                                                     |  ('Prose/Lyrical', 0.0)                       |                                             |                                                |  ('Prose/Lyrical', 0.0)                 |
|  ('A14 (academic)', 0.0)            |                                       |                                                       |                                                 |                                               |                                                |                                           |

#### Precision, recall and F1 scores using domain information as a signal of a "true label"

We used the most frequent label predicted on the domain as the "true label". Biggest values for each metric are in bold.

| Classifier (no. of labels)   |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| FTD-X-GENRE (7)  |       **0.57** |       0.76 |              **0.51** |           **0.67** |
| GINCO-X-GENRE (9)|       0.53 |       0.68 |              0.49 |           0.58 |
| CORE-X-GENRE  (9)|       0.53 |       0.65 |              0.5  |           0.59 |
| FTD (10)          |       0.52 |       0.74 |              0.46 |           0.62 |
| CORE  (9)        |       0.51 |       **0.78** |              0.45 |           0.63 |
| X-GENRE (9)      |       0.51 |       0.66 |              0.49 |           0.57 |
| GINCO  (9)       |       0.49 |       0.64 |              0.47 |           0.55 |

#### Comparison of X-GENRE classifier's performance based on X-GENRE majority label

I calculated the evaluation metrics for the X-GENRE classifiers (classifiers which use the X-GENRE schema) by taking the majority label (label predicted by most of the classifiers) as the "y_true" label. If there was a tie (more than 1 most common label), I randomly chose the majority label out of them.

Ties occurred in 11% of instances:

|     |   X-GENRE-majority-label-tie |
|:----|-----------------------------:|
| no  |                     0.889311 |
| yes |                     0.110689 |

The distribution of the majority X-GENRE predictions:

|                         |   X-GENRE-majority-label |
|:------------------------|-------------------------:|
| Promotion               |               0.463037   |
| Information/Explanation |               0.191009   |
| News                    |               0.120779   |
| Opinion/Argumentation   |               0.086014   |
| Instruction             |               0.0792208  |
| Other                   |               0.0225774  |
| Legal                   |               0.0167832  |
| Forum                   |               0.0138861  |
| Prose/Lyrical           |               0.00669331 |

Results:

| Classifier (labels)   |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| X-GENRE  (9)     |       **0.84** |       **0.88** |              **0.84** |           **0.85** |
| GINCO-X-GENRE (9) |       0.73 |       0.86 |              0.83 |           0.74 |
| FTD-X-GENRE (7)  |       0.68 |       0.74 |              0.76 |           0.68 |
| CORE-X-GENRE (9) |       0.48 |       0.53 |              0.38 |           0.75 |

#### Comparison of X-GENRE classifier agreement

I used the predictions of one classifier as y_true, and the predictions of the other as y_pred. I did it in both directions, just to check how the results change.
FTD-X-GENRE has less labels than the other (7, instead of 9), so whenever this classifier was in the pair, I used 7 labels for calculation of the evaluation metrics.

A problem: CORE-X-GENRE didn't predict Promotion to any of the instances - when calculating macro and micro F1, this affected the results - metrics for when CORE-X-GENRE labels are used as the list of labels (when its predictions are used as y_pred) are different than when the other classifier is used to create a list of labels (when its predictions are used as y_pred - except in the case of FTD-X-GENRE, which is always used for the list of labels, because it has less labels).

| Classifier as y_true   | Classifier as y_pred   |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:-----------------------|:-----------------------|-----------:|-----------:|------------------:|---------------:|
| GINCO-X-GENRE          | X-GENRE                |       **0.67** |       **0.79** |              0.64 |           **0.76** |
| X-GENRE                | GINCO-X-GENRE          |       **0.67** |       **0.79** |              **0.76** |           0.64 |
| FTD-X-GENRE            | X-GENRE                |       0.6  |       0.66 |              0.62 |           0.67 |
| X-GENRE                | FTD-X-GENRE            |       0.6  |       0.66 |              0.67 |           0.62 |
| GINCO-X-GENRE          | FTD-X-GENRE            |       0.53 |       0.69 |              0.57 |           0.65 |
| FTD-X-GENRE            | GINCO-X-GENRE          |       0.53 |       0.69 |              0.65 |           0.57 |
| X-GENRE                | CORE-X-GENRE           |       0.4  |       0.47 |              0.33 |           0.67 |
| CORE-X-GENRE           | X-GENRE                |       0.31 |       0.37 |              0.52 |           0.26 |
| CORE-X-GENRE           | GINCO-X-GENRE          |       0.27 |       0.32 |              0.51 |           0.22 |
| GINCO-X-GENRE          | CORE-X-GENRE           |       0.34 |       0.42 |              0.28 |           0.65 |
| CORE-X-GENRE           | FTD-X-GENRE            |       0.27 |       0.24 |              0.5  |           0.18 |
| FTD-X-GENRE            | CORE-X-GENRE           |       0.27 |       0.24 |              0.18 |           0.5  |

Based on the results, GINCO-X-GENRE and X-GENRE match the most, followed by FTD-X-GENRE and X-GENRE. On Micro F1 level, FTD-X-GENRE and GINCO-X-GENRE even outperform the combination of FTD-X-GENRE and X-GENRE (mostly because they both predict "Promotion" most frequently). CORE-X-GENRE matches the worst with all other. The worst agreement is between FTD-X-GENRE and CORE-X-GENRE.

![](figures/X-GENRE-comparison/Classifier-comparison-GINCO-X-GENRE-cm.png)

![](figures/X-GENRE-comparison/Classifier-comparison-GINCO-X-GENRE-report.png)

![](figures/X-GENRE-comparison/Classifier-comparison-FTD-X-GENRE-cm.png)

![](figures/X-GENRE-comparison/Classifier-comparison-FTD-X-GENRE-report.png)

![](figures/X-GENRE-comparison/Classifier-comparison-GINCO-FTD-cm.png)

![](figures/X-GENRE-comparison/Classifier-comparison-GINCO-FTD-report.png)

![](figures/X-GENRE-comparison/Classifier-comparison-CORE-FTD-cm.png)

![](figures/X-GENRE-comparison/Classifier-comparison-CORE-X-GENRE-cm.png)

#### Comparing schemata with Apriori algorithm

I had another idea that we could analyse which labels occur together using the apriori algorithm which is used in association rule learning.

Main concepts:

- Support: It measures the number of times a particular item or combination of items occur in a dataset out of the number of all instances.
`Support(pair) = frequency_of_pair/no_of_instances`

- Confidence: It measures how likely the pair will occur given they the left value has been predicted - number of times both have been predicted together divided by a number of time the left value has been predicted (-> if the left value occurs very often, the confidence will be smaller)
`Confidence(pair (based of occurrence of left value)) = frequency_of_pair/frequency of left value`

- Lift: A lift is a metric that determines the strength of association between the best rules. It is obtained by taking confidence (based on the frequency of the left value and right value) dand diving it with support (for right value). 
`Lift(pair (based on occurrence of left value)) = Confidence(based on occurrence of left value)/Support(right value)}`

This means that if left value is very frequent -> confidence is smaller, if the pair does not occur together very often, it won't reach the confidence limit. The frequency of right value does not impact the confidence, but it does impact the lift. For the same number of occurences of the pair, the lift is higher if one of them is infrequent than if both were frequent. The bigger support (frequency of the pair occuring together), the bigger is lift.

Parameters used: min_support=0.01, min_confidence=0.50, min_lift=1.0

Comparison: FTD with GINCO (very similar results also with GINCO-X-GENRE)

| Left_Hand_Side         | Right_Hand_Side                |   Support |   Confidence |     Lift |
|:-----------------------|:-------------------------------|----------:|-------------:|---------:|
| FTD: A7 (instruction)  | GINCO: Instruction             | 0.0373626 |     0.733333 | 10.238   |
| FTD: A8 (news)         | GINCO: News/Reporting          | 0.0324675 |     0.788835 |  7.03138 |
| FTD: A11 (personal)    | GINCO: Opinion/Argumentation   | 0.0241758 |     0.75625  |  6.66966 |
| FTD: A16 (information) | GINCO: Information/Explanation | 0.0748252 |     0.637447 |  4.54152 |
| FTD: A12 (promotion)   | GINCO: Promotion               | 0.40979   |     0.659274 |  1.52762 |

Labels not matched: ['FTD: A1 (argumentative)', 'FTD: A17 (review)', 'FTD: A9 (legal)', 'FTD: A14 (academic)', 'FTD: A4 (fiction)', 'GINCO: List of Summaries/Excerpts', 'GINCO: Other', 'GINCO: Legal/Regulation', 'GINCO: Forum']

--> Some labels not frequent enough (support too small) to be relevant for the target dataset (Legal/Regulation, Forum, academic, fiction) ... Some labels do not match with a label in the other schema consistently - argumentative, review, list of summaries/excerpts

Comparison: FTD with CORE
| Left_Hand_Side                              | Right_Hand_Side                             |   Support |   Confidence |    Lift |
|:--------------------------------------------|:--------------------------------------------|----------:|-------------:|--------:|
| FTD: A8 (news)                              | CORE: Narrative                             | 0.0305694 |     0.742718 | 7.67246 |
| FTD: A11 (personal)                         | CORE: Narrative                             | 0.0217782 |     0.68125  | 7.03747 |
| CORE: Informational Persuasion              | FTD: A12 (promotion)                        | 0.118282  |     0.964955 | 1.55243 |
| FTD: A9 (legal)                             | CORE: Informational Description/Explanation | 0.0195804 |     1        | 1.49225 |
| FTD: A16 (information)                      | CORE: Informational Description/Explanation | 0.106993  |     0.911489 | 1.36017 |
| CORE: Informational Description/Explanation | FTD: A12 (promotion)                        | 0.454146  |     0.677698 | 1.09029 |

Labels not matched: ['FTD: A1 (argumentative)', 'FTD: A7 (instruction)', 'FTD: A17 (review)', 'FTD: A14 (academic)', 'FTD: A4 (fiction)', 'CORE: Opinion', 'CORE: Spoken', 'CORE: How-To/Instructional', 'CORE: Interactive Discussion', 'CORE: Lyrical']

--> Informational/Description matches with legal, information and promotion 

Comparison: FTD with CORE-X-GENRE
| Left_Hand_Side                        | Right_Hand_Side                       |   Support |   Confidence |    Lift |
|:--------------------------------------|:--------------------------------------|----------:|-------------:|--------:|
| FTD: A8 (news)                        | CORE-X-GENRE: News                    | 0.0378621 |     0.919903 | 7.35481 |
| FTD: A7 (instruction)                 | CORE-X-GENRE: Instruction             | 0.035964  |     0.705882 | 3.60504 |
| FTD: A9 (legal)                       | CORE-X-GENRE: Information/Explanation | 0.0190809 |     0.97449  | 2.19996 |
| FTD: A16 (information)                | CORE-X-GENRE: Information/Explanation | 0.0944056 |     0.804255 | 1.81565 |
| CORE-X-GENRE: Instruction             | FTD: A12 (promotion)                  | 0.147852  |     0.755102 | 1.21481 |
| CORE-X-GENRE: Opinion/Argumentation   | FTD: A12 (promotion)                  | 0.0912088 |     0.648898 | 1.04395 |
| CORE-X-GENRE: Forum                   | FTD: A12 (promotion)                  | 0.0353646 |     0.641304 | 1.03174 |
| CORE-X-GENRE: Information/Explanation | FTD: A12 (promotion)                  | 0.283217  |     0.639378 | 1.02864 |

Labels not matched: ['FTD: A1 (argumentative)', 'FTD: A17 (review)', 'FTD: A11 (personal)', 'FTD: A14 (academic)', 'FTD: A4 (fiction)', 'CORE-X-GENRE: Prose/Lyrical', 'CORE-X-GENRE: Other']

Comparison: FTD with FTD-X-GENRE
| Left_Hand_Side                       | Right_Hand_Side        |   Support |   Confidence |     Lift |
|:-------------------------------------|:-----------------------|----------:|-------------:|---------:|
| FTD-X-GENRE: Legal                   | FTD: A9 (legal)        | 0.0188811 |     0.694853 | 35.4871  |
| FTD-X-GENRE: Opinion/Argumentation   | FTD: A11 (personal)    | 0.0223776 |     0.720257 | 22.5305  |
| FTD-X-GENRE: Instruction             | FTD: A7 (instruction)  | 0.041958  |     0.720412 | 14.1398  |
| FTD-X-GENRE: News                    | FTD: A8 (news)         | 0.036963  |     0.581761 | 14.1345  |
| FTD-X-GENRE: Information/Explanation | FTD: A16 (information) | 0.0994006 |     0.630545 |  5.37171 |

Labels not matched: ['FTD: A12 (promotion)', 'FTD: A1 (argumentative)', 'FTD: A17 (review)', 'FTD: A14 (academic)', 'FTD: A4 (fiction)', 'FTD-X-GENRE: Promotion', 'FTD-X-GENRE: Prose/Lyrical']

Comparison: FTD with X-GENRE
| Left_Hand_Side         | Right_Hand_Side                  |   Support |   Confidence |     Lift |
|:-----------------------|:---------------------------------|----------:|-------------:|---------:|
| FTD: A9 (legal)        | X-GENRE: Legal                   | 0.0143856 |     0.734694 | 38.5041  |
| FTD: A7 (instruction)  | X-GENRE: Instruction             | 0.0386613 |     0.758824 |  9.6639  |
| FTD: A11 (personal)    | X-GENRE: Opinion/Argumentation   | 0.0267732 |     0.8375   |  7.41892 |
| FTD: A8 (news)         | X-GENRE: News                    | 0.0397602 |     0.966019 |  7.19483 |
| FTD: A16 (information) | X-GENRE: Information/Explanation | 0.0806194 |     0.686809 |  3.83006 |
| FTD: A12 (promotion)   | X-GENRE: Promotion               | 0.407692  |     0.655898 |  1.53544 |

Labels not matched: ['FTD: A1 (argumentative)', 'FTD: A17 (review)', 'FTD: A14 (academic)', 'FTD: A4 (fiction)', 'X-GENRE: Other', 'X-GENRE: Prose/Lyrical', 'X-GENRE: Forum']

Comparison: GINCO with CORE
| Left_Hand_Side                    | Right_Hand_Side                             |   Support |   Confidence |    Lift |
|:----------------------------------|:--------------------------------------------|----------:|-------------:|--------:|
| CORE: How-To/Instructional        | GINCO: Instruction                          | 0.033966  |     0.65764  | 9.18128 |
| CORE: Opinion                     | GINCO: Opinion/Argumentation                | 0.0248751 |     0.680328 | 6.00007 |
| CORE: Informational Persuasion    | GINCO: Promotion                            | 0.11039   |     0.90057  | 2.08674 |
| GINCO: Information/Explanation    | CORE: Informational Description/Explanation | 0.136364  |     0.97153  | 1.44976 |
| GINCO: List of Summaries/Excerpts | CORE: Informational Description/Explanation | 0.0607393 |     0.73697  | 1.09974 |
| GINCO: Other                      | CORE: Informational Description/Explanation | 0.0177822 |     0.706349 | 1.05405 |
| GINCO: Promotion                  | CORE: Informational Description/Explanation | 0.304296  |     0.705093 | 1.05217 |

Labels not matched: ['GINCO: News/Reporting', 'GINCO: Legal/Regulation', 'GINCO: Forum', 'CORE: Narrative', 'CORE: Spoken', 'CORE: Interactive Discussion', 'CORE: Lyrical']

Comparison: GINCO with FTD-X-GENRE
| Left_Hand_Side                       | Right_Hand_Side                |   Support |   Confidence |    Lift |
|:-------------------------------------|:-------------------------------|----------:|-------------:|--------:|
| FTD-X-GENRE: Instruction             | GINCO: Instruction             | 0.0396603 |     0.680961 | 9.50686 |
| FTD-X-GENRE: News                    | GINCO: News/Reporting          | 0.046953  |     0.738994 | 6.58711 |
| FTD-X-GENRE: Opinion/Argumentation   | GINCO: Opinion/Argumentation   | 0.0214785 |     0.691318 | 6.097   |
| FTD-X-GENRE: Information/Explanation | GINCO: Information/Explanation | 0.0865135 |     0.548796 | 3.90993 |
| FTD-X-GENRE: Promotion               | GINCO: Promotion               | 0.40979   |     0.625973 | 1.45046 |
| GINCO: List of Summaries/Excerpts    | FTD-X-GENRE: Promotion         | 0.0609391 |     0.739394 | 1.12946 |

Labels not matched: ['GINCO: Other', 'GINCO: Legal/Regulation', 'GINCO: Forum', 'FTD-X-GENRE: Legal', 'FTD-X-GENRE: Prose/Lyrical']

Comparison: GINCO with CORE-X-GENRE
| Left_Hand_Side                 | Right_Hand_Side                       |   Support |   Confidence |     Lift |
|:-------------------------------|:--------------------------------------|----------:|-------------:|---------:|
| GINCO: Forum                   | CORE-X-GENRE: Forum                   | 0.010989  |     0.763889 | 13.8524  |
| CORE-X-GENRE: Prose/Lyrical    | GINCO: Opinion/Argumentation          | 0.012987  |     0.646766 |  5.70408 |
| CORE-X-GENRE: News             | GINCO: News/Reporting                 | 0.0647353 |     0.517572 |  4.61344 |
| GINCO: Instruction             | CORE-X-GENRE: Instruction             | 0.0555445 |     0.775453 |  3.96035 |
| GINCO: Information/Explanation | CORE-X-GENRE: Information/Explanation | 0.126074  |     0.898221 |  2.02778 |
| CORE-X-GENRE: Instruction      | GINCO: Promotion                      | 0.113686  |     0.580612 |  1.34535 |

Labels not matched: ['GINCO: List of Summaries/Excerpts', 'GINCO: Other', 'GINCO: Legal/Regulation', 'CORE-X-GENRE: Opinion/Argumentation', 'CORE-X-GENRE: Other']

Comparison: CORE with CORE-X-GENRE
| Left_Hand_Side                        | Right_Hand_Side                             |   Support |   Confidence |     Lift |
|:--------------------------------------|:--------------------------------------------|----------:|-------------:|---------:|
| CORE: Interactive Discussion          | CORE-X-GENRE: Forum                         | 0.0105894 |     0.84127  | 15.2556  |
| CORE: How-To/Instructional            | CORE-X-GENRE: Instruction                   | 0.0434565 |     0.841393 |  4.29711 |
| CORE: Narrative                       | CORE-X-GENRE: News                          | 0.0503497 |     0.520124 |  4.1585  |
| CORE-X-GENRE: Information/Explanation | CORE: Informational Description/Explanation | 0.381718  |     0.86175  |  1.28594 |

Labels not matched: ['CORE: Informational Persuasion', 'CORE: Opinion', 'CORE: Spoken', 'CORE: Lyrical', 'CORE-X-GENRE: Opinion/Argumentation', 'CORE-X-GENRE: Prose/Lyrical', 'CORE-X-GENRE: Other']

Comparison: CORE with X-GENRE
| Left_Hand_Side                   | Right_Hand_Side                             |   Support |   Confidence |    Lift |
|:---------------------------------|:--------------------------------------------|----------:|-------------:|--------:|
| CORE: How-To/Instructional       | X-GENRE: Instruction                        | 0.0400599 |     0.775629 | 9.87792 |
| CORE: Opinion                    | X-GENRE: Opinion/Argumentation              | 0.0235764 |     0.644809 | 5.71198 |
| CORE: Informational Persuasion   | X-GENRE: Promotion                          | 0.11029   |     0.899756 | 2.1063  |
| X-GENRE: Legal                   | CORE: Informational Description/Explanation | 0.0190809 |     1        | 1.49225 |
| X-GENRE: Information/Explanation | CORE: Informational Description/Explanation | 0.170629  |     0.951532 | 1.41992 |
| X-GENRE: Promotion               | CORE: Informational Description/Explanation | 0.301299  |     0.705332 | 1.05253 |

Labels not matched: ['CORE: Narrative', 'CORE: Spoken', 'CORE: Interactive Discussion', 'CORE: Lyrical', 'X-GENRE: Other', 'X-GENRE: News', 'X-GENRE: Prose/Lyrical', 'X-GENRE: Forum']

Comparison: GINCO-X-GENRE with FTD-X-GENRE
| Left_Hand_Side                       | Right_Hand_Side                        |   Support |   Confidence |    Lift |
|:-------------------------------------|:---------------------------------------|----------:|-------------:|--------:|
| FTD-X-GENRE: Instruction             | GINCO-X-GENRE: Instruction             | 0.0393606 |     0.675815 | 9.3309  |
| FTD-X-GENRE: Opinion/Argumentation   | GINCO-X-GENRE: Opinion/Argumentation   | 0.0204795 |     0.659164 | 7.29893 |
| FTD-X-GENRE: News                    | GINCO-X-GENRE: News                    | 0.0549451 |     0.86478  | 6.90858 |
| FTD-X-GENRE: Information/Explanation | GINCO-X-GENRE: Information/Explanation | 0.0916084 |     0.581115 | 3.97062 |
| FTD-X-GENRE: Promotion               | GINCO-X-GENRE: Promotion               | 0.451948  |     0.690371 | 1.43314 |

Labels not matched: ['GINCO-X-GENRE: Other', 'GINCO-X-GENRE: Legal', 'GINCO-X-GENRE: Forum', 'GINCO-X-GENRE: Prose/Lyrical', 'FTD-X-GENRE: Legal', 'FTD-X-GENRE: Prose/Lyrical']

Comparison: GINCO-X-GENRE with CORE-X-GENRE
| Left_Hand_Side                         | Right_Hand_Side                       |   Support |   Confidence |     Lift |
|:---------------------------------------|:--------------------------------------|----------:|-------------:|---------:|
| GINCO-X-GENRE: Forum                   | CORE-X-GENRE: Forum                   | 0.0110889 |     0.816176 | 14.8006  |
| CORE-X-GENRE: Prose/Lyrical            | GINCO-X-GENRE: Opinion/Argumentation  | 0.0132867 |     0.661692 |  7.32692 |
| CORE-X-GENRE: News                     | GINCO-X-GENRE: News                   | 0.0782218 |     0.625399 |  4.99621 |
| GINCO-X-GENRE: Instruction             | CORE-X-GENRE: Instruction             | 0.0526474 |     0.726897 |  3.71236 |
| GINCO-X-GENRE: Information/Explanation | CORE-X-GENRE: Information/Explanation | 0.129371  |     0.883959 |  1.99559 |
| CORE-X-GENRE: Instruction              | GINCO-X-GENRE: Promotion              | 0.125175  |     0.639286 |  1.32709 |
| CORE-X-GENRE: Opinion/Argumentation    | GINCO-X-GENRE: Promotion              | 0.0745255 |     0.530206 |  1.10066 |

Labels not matched: ['GINCO-X-GENRE: Other', 'GINCO-X-GENRE: Legal', 'GINCO-X-GENRE: Prose/Lyrical', 'CORE-X-GENRE: Other']

Comparison: FTD-X-GENRE with GINCO-X-GENRE
| Left_Hand_Side                       | Right_Hand_Side                        |   Support |   Confidence |    Lift |
|:-------------------------------------|:---------------------------------------|----------:|-------------:|--------:|
| FTD-X-GENRE: Instruction             | GINCO-X-GENRE: Instruction             | 0.0393606 |     0.675815 | 9.3309  |
| FTD-X-GENRE: Opinion/Argumentation   | GINCO-X-GENRE: Opinion/Argumentation   | 0.0204795 |     0.659164 | 7.29893 |
| FTD-X-GENRE: News                    | GINCO-X-GENRE: News                    | 0.0549451 |     0.86478  | 6.90858 |
| FTD-X-GENRE: Information/Explanation | GINCO-X-GENRE: Information/Explanation | 0.0916084 |     0.581115 | 3.97062 |
| FTD-X-GENRE: Promotion               | GINCO-X-GENRE: Promotion               | 0.451948  |     0.690371 | 1.43314 |

Labels not matched: ['FTD-X-GENRE: Legal', 'FTD-X-GENRE: Prose/Lyrical', 'GINCO-X-GENRE: Other', 'GINCO-X-GENRE: Legal', 'GINCO-X-GENRE: Forum', 'GINCO-X-GENRE: Prose/Lyrical']

Comparison: FTD-X-GENRE with CORE-X-GENRE
| Left_Hand_Side                       | Right_Hand_Side                       |   Support |   Confidence |    Lift |
|:-------------------------------------|:--------------------------------------|----------:|-------------:|--------:|
| FTD-X-GENRE: News                    | CORE-X-GENRE: News                    | 0.0513487 |     0.808176 | 6.46154 |
| FTD-X-GENRE: Instruction             | CORE-X-GENRE: Instruction             | 0.042957  |     0.737564 | 3.76685 |
| FTD-X-GENRE: Legal                   | CORE-X-GENRE: Information/Explanation | 0.0255744 |     0.941176 | 2.12476 |
| FTD-X-GENRE: Information/Explanation | CORE-X-GENRE: Information/Explanation | 0.12028   |     0.762991 | 1.72249 |
| CORE-X-GENRE: Opinion/Argumentation  | FTD-X-GENRE: Promotion                | 0.11039   |     0.785359 | 1.19967 |
| CORE-X-GENRE: Other                  | FTD-X-GENRE: Promotion                | 0.0148851 |     0.730392 | 1.11571 |
| CORE-X-GENRE: Instruction            | FTD-X-GENRE: Promotion                | 0.141359  |     0.721939 | 1.10279 |
| CORE-X-GENRE: Forum                  | FTD-X-GENRE: Promotion                | 0.0376623 |     0.682971 | 1.04327 |

Labels not matched: ['FTD-X-GENRE: Opinion/Argumentation', 'FTD-X-GENRE: Prose/Lyrical', 'CORE-X-GENRE: Prose/Lyrical']

Comparison: FTD-X-GENRE with X-GENRE
| Left_Hand_Side                       | Right_Hand_Side                  |   Support |   Confidence |     Lift |
|:-------------------------------------|:---------------------------------|----------:|-------------:|---------:|
| FTD-X-GENRE: Legal                   | X-GENRE: Legal                   | 0.0170829 |     0.628676 | 32.9479  |
| FTD-X-GENRE: Instruction             | X-GENRE: Instruction             | 0.0432567 |     0.74271  |  9.45869 |
| FTD-X-GENRE: Opinion/Argumentation   | X-GENRE: Opinion/Argumentation   | 0.0235764 |     0.758842 |  6.72214 |
| FTD-X-GENRE: News                    | X-GENRE: News                    | 0.0549451 |     0.86478  |  6.44081 |
| FTD-X-GENRE: Information/Explanation | X-GENRE: Information/Explanation | 0.0975025 |     0.618504 |  3.44915 |
| FTD-X-GENRE: Promotion               | X-GENRE: Promotion               | 0.40999   |     0.626278 |  1.4661  |
| X-GENRE: Other                       | FTD-X-GENRE: Promotion           | 0.0243756 |     0.743902 |  1.13634 |

Labels not matched: ['FTD-X-GENRE: Prose/Lyrical', 'X-GENRE: Prose/Lyrical', 'X-GENRE: Forum']

Comparison: CORE-X-GENRE with X-GENRE
| Left_Hand_Side                   | Right_Hand_Side                       |   Support |   Confidence |     Lift |
|:---------------------------------|:--------------------------------------|----------:|-------------:|---------:|
| X-GENRE: Forum                   | CORE-X-GENRE: Forum                   | 0.0103896 |     0.912281 | 16.5434  |
| CORE-X-GENRE: Prose/Lyrical      | X-GENRE: Opinion/Argumentation        | 0.0118881 |     0.59204  |  5.24453 |
| CORE-X-GENRE: News               | X-GENRE: News                         | 0.079021  |     0.631789 |  4.70551 |
| X-GENRE: Instruction             | CORE-X-GENRE: Instruction             | 0.0636364 |     0.810433 |  4.13899 |
| X-GENRE: Legal                   | CORE-X-GENRE: Information/Explanation | 0.0175824 |     0.921466 |  2.08026 |
| X-GENRE: Information/Explanation | CORE-X-GENRE: Information/Explanation | 0.154745  |     0.862953 |  1.94816 |
| CORE-X-GENRE: Instruction        | X-GENRE: Promotion                    | 0.108891  |     0.556122 |  1.30187 |

Labels not matched: ['CORE-X-GENRE: Opinion/Argumentation', 'CORE-X-GENRE: Other', 'X-GENRE: Other', 'X-GENRE: Prose/Lyrical']

# Comparison of classifiers on the second sample of MaCoCu-sl and on MaCoCu-mk

I re-did all experiments on a new sample of MaCoCu-sl, where I chosen 500 random domains above and 500 random domains below the median instead of the domains around the median (as I did in the first sample.)

I then extended the experiments to the Macedonian dataset - MaCoCu-mk - as well, following the same steps as with the MaCoCu-sl.

#### Comparison of confidence of the predictions

MaCoCu-sl
- no big differences with sample1

| classifier    |   min |   median |   max |
|:--------------|------:|---------:|------:|
| X-GENRE       |  0.33 |     1    |  1    |
| GINCO-X-GENRE |  0.26 |     0.99 |  0.99 |
| GINCO         |  0.25 |     0.94 |  0.98 |
| CORE          |  0.25 |     0.86 |  0.99 |
| FTD-X-GENRE   |  0.19 |     0.86 |  0.97 |
| FTD           |  0.16 |     0.78 |  0.96 |
| CORE-X-GENRE  |  0.15 |     0.54 |  0.95 |

MaCoCu-mk

| classifier    |   min |   median |   max |
|:--------------|------:|---------:|------:|
| X-GENRE       |  0.32 |     1    |  1    |
| GINCO-X-GENRE |  0.28 |     0.98 |  0.99 |
| GINCO         |  0.24 |     0.95 |  0.98 |
| CORE          |  0.24 |     0.91 |  0.99 |
| FTD-X-GENRE   |  0.22 |     0.83 |  0.97 |
| FTD           |  0.16 |     0.77 |  0.96 |
| CORE-X-GENRE  |  0.15 |     0.64 |  0.95 |

The median confidence of CORE and CORE-X-GENRE is bigger on MaCoCu-mk (5-10 points).

#### Most frequent label (on instance level) per classifier

MaCoCu-sl:
- The frequency of the most frequent label between sample1 and sample2 of MaCoCu-sl changed only slightly (up to 5 percent) - it is slightly less frequent now.

| classifier    | most frequent label                   |   frequency |
|:--------------|:--------------------------------------|------------:|
| FTD           | A12 (promotion)                       |        0.59 |
| GINCO         | Promotion                             |        0.4  |
| CORE          | Informational Description/Explanation |        0.65 |
| GINCO-X-GENRE | Promotion                             |        0.45 |
| FTD-X-GENRE   | Promotion                             |        **0.63** |
| CORE-X-GENRE  | Information/Explanation               |        0.41 |
| X-GENRE       | Promotion                             |        0.39 |

MaCoCu-mk:

| classifier    | most frequent label                   |   frequency |
|:--------------|:--------------------------------------|------------:|
| FTD           | A12 (promotion)                       |        0.42 |
| GINCO         | News/Reporting                        |        0.31 |
| CORE          | Informational Description/Explanation |        0.58 |
| GINCO-X-GENRE | News                                  |        0.31 |
| FTD-X-GENRE   | Promotion                             |        0.44 |
| CORE-X-GENRE  | Information/Explanation               |        0.36 |
| X-GENRE       | News                                  |        0.34 |

Slovene and Macedonian datasets differ in terms of which label was shown to be the most frequent in some of the classifiers. FTD and FTD-X-GENRE still predict Promotion as the most frequent label, but with a smaller frequency (17-19 points smaller). According to GINCO, X-GENRE and GINCO-X-GENRE, the most frequent label in MK (= MaCoCu-mk) is News (in SL, it is Promotion), while CORE still predicts Information/Explanation as the most frequent label, but it is a bit less frequently predicted in MK (5-7 points less frequent).

It seems that the label distribution is a bit less ruled by just one label as in SL (=MaCoCu-sl).

<!-- 

#### Comparison of label distribution (instance level)

MaCoCu-sl:
- Label distribution between sample1 and sample2 of MaCoCu-sl remains more or less the same (very small differences between the samples).

| label_distribution_FTD          | label_distribution_GINCO                | label_distribution_CORE                            | label_distribution_GINCO-X-GENRE  | label_distribution_FTD-X-GENRE       | label_distribution_CORE-X-GENRE      | label_distribution_X-GENRE           |
|------------------------------------|--------------------------------------------|-------------------------------------------------------|--------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| ('A12 (promotion)', 0.59)       | ('Promotion', 0.4)                      | ('Informational Description/Explanation', 0.65) | ('Promotion', 0.45)               | ('Promotion', **0.63**)                  | ('Information/Explanation', 0.41) | ('Promotion', 0.39)                  |
| ('A16 (information)', 0.13)     | ('Information/Explanation', 0.14)       | ('Informational Persuasion', 0.12)                 | ('Information/Explanation', 0.15) | ('Information/Explanation', 0.17) | ('Instruction', 0.2)                 | ('Information/Explanation', 0.18) |
| ('A1 (argumentative)', 0.07) | ('Opinion/Argumentation', 0.12)         | ('Narrative', 0.11)                                | ('News', 0.14)                    | ('News', 0.08)                       | ('Opinion/Argumentation', 0.14)      | ('News', 0.15)                       |
| ('A8 (news)', 0.05)             | ('News/Reporting', 0.12)                | ('How-To/Instructional', 0.05)                     | ('Opinion/Argumentation', 0.1)    | ('Instruction', 0.06)                | ('News', 0.13)                       | ('Opinion/Argumentation', 0.12)      |
| ('A7 (instruction)', 0.05)      | ('List of Summaries/Excerpts', 0.08) | ('Opinion', 0.04)                                  | ('Instruction', 0.08)             | ('Opinion/Argumentation', 0.03)      | ('Forum', 0.07)                      | ('Instruction', 0.09)                |
| ('A17 (review)', 0.05)          | ('Instruction', 0.07)                   | ('Interactive Discussion', 0.02)                   | ('Other', 0.06)                   | ('Legal', 0.02)                      | ('Prose/Lyrical', 0.02)              | ('Other', 0.03)                      |
| ('A11 (personal)', 0.03)        | ('Other', 0.03)                         | ('Spoken', 0.01)                                   | ('Forum', 0.02)                   | ('Prose/Lyrical', 0.01)              | ('Other', 0.02)                      | ('Forum', 0.02)                      |
| ('A9 (legal)', 0.02)            | ('Forum', 0.02)                         | ('Lyrical', 0.0)                                   | ('Legal', 0.01)                   |                                      |                                         | ('Legal', 0.02)                      |
| ('A4 (fiction)', 0.01)          | ('Legal/Regulation', 0.01)              |                                                    | ('Prose/Lyrical', 0.0)            |                                      |                                         | ('Prose/Lyrical', 0.01)              |
| ('A14 (academic)', 0.0)         |                                         |                                                       |                                      |                                         |                                         |                                         |

 -->

#### Comparison of frequency of prediction of the most frequent label per domain

MaCoCu-sl: distributions in sample2 are very similar to the sample1.

![](Comparison-of-distribution-in-domains-MaCoCu-sl-histogram-sample2.png)

MaCoCu-mk:

![](Comparison-of-distribution-in-domains-MaCoCu-mk-histogram.png)

We can see that CORE still has the highest frequency of prediction of the most frequent label per domain, but all other classifiers are now much more similar to each other.

MaCoCu-sl:

![](Comparison-of-distribution-in-domains-MaCoCu-sl-subplots-sample2.png)

MaCoCu-mk:

![](Comparison-of-distribution-in-domains-MaCoCu-mk-subplots.png)

MaCoCu-sl:

![](Comparison-of-distribution-in-domains-MaCoCu-sl-KDE-sample2.png)

MaCoCu-mk:
![](Comparison-of-distribution-in-domains-MaCoCu-mk-KDE.png)

#### Comparison of label distribution on the domain level

When comparing the two samples of MaCoCu-sl, frequency of the most frequent label on domain level is slightly smaller for all classifiers, but only up to 5 points. Otherwise, the distributions are very similar.

Table shows in how many of the domains a label is the most frequent label in the domain. The values in the table are percentages.

MaCoCu-sl:

| most frequent label in domain: FTD | most frequent label in domain: GINCO | most frequent label in domain: CORE                | most frequent label in domain: GINCO-X-GENRE | most frequent label in domain: FTD-X-GENRE | most frequent label in domain: CORE-X-GENRE | most frequent label in domain: X-GENRE |
|---------------------------------------|-----------------------------------------|-------------------------------------------------------|-------------------------------------------------|-----------------------------------------------|------------------------------------------------|-------------------------------------------|
| ('A12 (promotion)', 0.7)           | ('Promotion', 0.5)                   | ('Informational Description/Explanation', 0.73) | ('Promotion', 0.53)                          | ('Promotion', 0.72)                        | ('Information/Explanation', 0.47)           | ('Promotion', 0.46)                    |
| ('A16 (information)', 0.09)        | ('Information/Explanation', 0.13)    | ('Informational Persuasion', 0.1)                  | ('News', 0.14)                               | ('Information/Explanation', 0.13)          | ('Instruction', 0.19)                       | ('Information/Explanation', 0.16)      |
| ('A1 (argumentative)', 0.06)       | ('News/Reporting', 0.13)             | ('Narrative', 0.09)                                | ('Information/Explanation', 0.13)            | ('News', 0.07)                             | ('Opinion/Argumentation', 0.13)             | ('News', 0.16)                         |
| ('A8 (news)', 0.05)                | ('Opinion/Argumentation', 0.11)      | ('How-To/Instructional', 0.04)                     | ('Opinion/Argumentation', 0.09)              | ('Instruction', 0.04)                      | ('News', 0.12)                              | ('Opinion/Argumentation', 0.1)         |
| ('A17 (review)', 0.04)             | ('Instruction', 0.05)                | ('Interactive Discussion', 0.02)                   | ('Instruction', 0.05)                        | ('Opinion/Argumentation', 0.03)            | ('Forum', 0.06)                             | ('Instruction', 0.07)                  |
| ('A7 (instruction)', 0.03)         | ('List of Summaries/Excerpts', 0.04) | ('Opinion', 0.02)                                  | ('Other', 0.03)                              | ('Legal', 0.01)                            | ('Prose/Lyrical', 0.02)                     | ('Forum', 0.03)                        |
| ('A11 (personal)', 0.02)           | ('Forum', 0.03)                      | ('Lyrical', 0.0)                                   | ('Forum', 0.02)                              | ('Prose/Lyrical', 0.01)                    | ('Other', 0.01)                             | ('Other', 0.01)                        |
| ('A9 (legal)', 0.01)               | ('Other', 0.01)                      | ('Spoken', 0.0)                                    | ('Legal', 0.0)                               |                                            |                                                | ('Legal', 0.01)                        |
| ('A4 (fiction)', 0.0)              | ('Legal/Regulation', 0.0)            |                                                    | ('Prose/Lyrical', 0.0)                       |                                            |                                                | ('Prose/Lyrical', 0.0)                 |
| ('A14 (academic)', 0.0)            |                                      |                                                       |                                                 |                                               |                                                |                                           |

MaCoCu-mk:

| most frequent label in domain: FTD | most frequent label in domain: GINCO | most frequent label in domain: CORE                | most frequent label in domain: GINCO-X-GENRE | most frequent label in domain: FTD-X-GENRE | most frequent label in domain: CORE-X-GENRE | most frequent label in domain: X-GENRE |
|---------------------------------------|-----------------------------------------|-------------------------------------------------------|-------------------------------------------------|-----------------------------------------------|------------------------------------------------|-------------------------------------------|
| ('A12 (promotion)', 0.5)           | ('News/Reporting', 0.41)             | ('Informational Description/Explanation', **0.63**) | ('News', 0.38)                               | ('Promotion', 0.48)                        | ('Information/Explanation', 0.41)           | ('News', 0.41)                         |
| ('A8 (news)', 0.22)                | ('Promotion', 0.26)                  | ('Narrative', 0.24)                                | ('Promotion', 0.33)                          | ('News', 0.3)                              | ('News', 0.34)                              | ('Promotion', 0.3)                     |
| ('A1 (argumentative)', 0.12)       | ('Information/Explanation', 0.17)    | ('Informational Persuasion', 0.06)                 | ('Information/Explanation', 0.17)            | ('Information/Explanation', 0.16)          | ('Instruction', 0.11)                       | ('Information/Explanation', 0.15)      |
| ('A16 (information)', 0.09)        | ('Opinion/Argumentation', 0.07)      | ('Opinion', 0.03)                                  | ('Opinion/Argumentation', 0.05)              | ('Instruction', 0.03)                      | ('Opinion/Argumentation', 0.09)             | ('Opinion/Argumentation', 0.06)        |
| ('A17 (review)', 0.03)             | ('List of Summaries/Excerpts', 0.04) | ('How-To/Instructional', 0.02)                     | ('Instruction', 0.03)                        | ('Legal', 0.02)                            | ('Forum', 0.03)                             | ('Instruction', 0.05)                  |
| ('A7 (instruction)', 0.03)         | ('Instruction', 0.03)                | ('Interactive Discussion', 0.01)                   | ('Other', 0.03)                              | ('Opinion/Argumentation', 0.02)            | ('Prose/Lyrical', 0.01)                     | ('Forum', 0.01)                        |
| ('A9 (legal)', 0.01)               | ('Forum', 0.01)                      | ('Lyrical', 0.0)                                   | ('Forum', 0.01)                              | ('Prose/Lyrical', 0.0)                     | ('Other', 0.0)                              | ('Legal', 0.01)                        |
| ('A4 (fiction)', 0.0)              | ('Other', 0.01)                      | ('Spoken', 0.0)                                    | ('Legal', 0.0)                               |                                            |                                                | ('Other', 0.01)                        |
| ('A14 (academic)', 0.0)            | ('Legal/Regulation', 0.0)            |                                                    |                                                 |                                               |                                                | ('Prose/Lyrical', 0.0)                 |
| ('A11 (personal)', 0.0)            |                                      |                                                       |                                                 |                                               |                                                |                                           |

#### Precision, recall and F1 scores using domain information as a signal of a "true label"

We used the most frequent label predicted on the domain as the "true label". Biggest values for each metric are in bold.

MaCoCu-sl:
- in sample1, the ranking based on Macro F1 was FTD-X-GENRE (0.57), GINCO-X-GENRE and CORE-X-GENRE (sharing the 2nd spot: 0.53), FTD (0.52), CORE and X-GENRE (0.51), GINCO (0.49).

- in sample2, CORE has a slightly higher Macro F1 score, ranking is: FTD-X-GENRE (0.58), CORE-X-GENRE (0.56), CORE (0.54), GINCO-X-GENRE (0.53), GINCO and X-GENRE (0.51), FTD (0.49). However, in both samples, all scores are pretty similar.

| Classifier    |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| FTD-X-GENRE   |       **0.58** |       0.75 |              **0.53** |           **0.65** |
| CORE-X-GENRE  |       0.56 |       0.65 |              **0.53** |           0.61 |
| CORE          |       0.54 |       **0.78** |              0.49 |           **0.65** |
| GINCO-X-GENRE |       0.53 |       0.67 |              0.52 |           0.57 |
| GINCO         |       0.51 |       0.64 |              0.49 |           0.58 |
| X-GENRE       |       0.51 |       0.66 |              0.49 |           0.6  |
| FTD           |       0.49 |       0.72 |              0.44 |           0.6  |

MaCoCu-mk:

| Classifier    |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| FTD-X-GENRE   |       **0.53** |       0.71 |              **0.5**  |           **0.62** |
| CORE          |       0.52 |       **0.77** |              0.48 |           **0.62** |
| GINCO         |       0.51 |       0.65 |              0.48 |           **0.62** |
| CORE-X-GENRE  |       0.51 |       0.67 |              0.48 |           0.59 |
| X-GENRE       |       0.5  |       0.68 |              0.47 |           0.58 |
| GINCO-X-GENRE |       0.47 |       0.66 |              0.45 |           0.56 |
| FTD           |       0.46 |       0.68 |              0.41 |           0.58 |

The scores are similar to the MaCoCu-sl and there are no big differences in the ranking order. The scores are slighlty smaller, I assume the reason for this is that the MaCoCu-mk is slightly more diverse in terms of the percentage of texts in the most frequent genre, and that is why just predicting one genre gave worse scores to the classifiers that are biased towards one genre (FTD prefers Promotion, CORE prefers Information/Explanation).

#### Comparison of X-GENRE classifier's performance based on X-GENRE majority label

I calculated the evaluation metrics for the X-GENRE classifiers (classifiers which use the X-GENRE schema) by taking the majority label (label predicted by most of the classifiers) as the "y_true" label. If there was a tie (more than 1 most common label), I randomly chose the majority label out of them. There were around 10 % of ties in all datasets.

**Distribution of majority X-GENRE labels in MaCoCu-sl**

Distribution is very similar to the sample1.

|                         |   X-GENRE-majority-label |
|:------------------------|-------------------------:|
| Promotion               |                   0.4265 |
| Information/Explanation |                   0.1902 |
| News                    |                   0.1347 |
| Opinion/Argumentation   |                   0.0918 |
| Instruction             |                   0.0861 |
| Forum                   |                   0.0252 |
| Other                   |                   0.0243 |
| Legal                   |                   0.0141 |
| Prose/Lyrical           |                   0.0071 |

**Distribution of majority X-GENRE labels in MaCoCu-mk**

|                         |   X-GENRE-majority-label |
|:------------------------|-------------------------:|
| News                    |                   0.3261 |
| Promotion               |                   0.2827 |
| Information/Explanation |                   0.2103 |
| Instruction             |                   0.0617 |
| Opinion/Argumentation   |                   0.0586 |
| Legal                   |                   0.0202 |
| Other                   |                   0.0201 |
| Forum                   |                   0.0121 |
| Prose/Lyrical           |                   0.0082 |

In contrast to SL, in MK, the most frequent label is News, not Promotion. The second most frequent labels is Promotion, while in the SL, it is Information/Explanation.

**Results - MaCoCu-sl**

The ranking remains the same as in sample1, only the values are mostly a couple of points slightly higher. No big differences between the samples were observed, though.

| Classifier    |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| X-GENRE       |       **0.87** |       **0.89** |              **0.86** |           **0.89** |
| GINCO-X-GENRE |       0.73 |       0.86 |              0.85 |           0.73 |
| FTD-X-GENRE   |       0.67 |       0.73 |              0.74 |           0.69 |
| CORE-X-GENRE  |       0.5  |       0.56 |              0.41 |           0.74 |

**Results - MaCoCu-mk**

| Classifier    |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:--------------|-----------:|-----------:|------------------:|---------------:|
| X-GENRE       |       **0.85** |       **0.9**  |              **0.83** |           **0.87** |
| FTD-X-GENRE   |       0.7  |       0.78 |              0.75 |           0.72 |
| GINCO-X-GENRE |       0.69 |       0.85 |              0.81 |           0.7  |
| CORE-X-GENRE  |       0.57 |       0.67 |              0.48 |           **0.79** |

The classifiers seem to be more comparable based on the results in MaCoCu-mk sample in all cases except in the case of GINCO-X-GENRE where the scores are slightly lower. However, in general, scores are quite similar. X-GENRE remains the most similar to the majority results.

#### Comparison of X-GENRE classifier agreement

I used the predictions of one classifier as y_true, and the predictions of the other as y_pred. I did it in both directions, just to check how the results change.
FTD-X-GENRE has less labels than the other (7, instead of 9), so whenever this classifier was in the pair, I used 7 labels for calculation of the evaluation metrics.

**MaCoCu-sl**

The ranking order is the same as in sample1 and the results are very similar as well. CORE-X-GENRE now has a bit higher scores when in combination with GINCO-X-GENRE and X-GENRE (up to 5 points bigger Macro F1).

| Classifier as y_true   | Classifier as y_pred   |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:-----------------------|:-----------------------|-----------:|-----------:|------------------:|---------------:|
| GINCO-X-GENRE          | X-GENRE                |       **0.67** |       ****0.79**** |              0.64 |           **0.79** |
| X-GENRE                | GINCO-X-GENRE          |       0.67 |       **0.79** |              ****0.79****|           0.64 |
| FTD-X-GENRE            | X-GENRE                |       0.6  |       0.66 |              **0.63** |           0.67 |
| X-GENRE                | FTD-X-GENRE            |       0.6  |       0.66 |              0.67 |           **0.63** |
| GINCO-X-GENRE          | FTD-X-GENRE            |       0.52 |       0.68 |              0.55 |           0.65 |
| FTD-X-GENRE            | GINCO-X-GENRE          |       0.52 |       0.68 |              0.65 |           0.55 |
| X-GENRE                | CORE-X-GENRE           |       0.45 |       0.51 |              0.38 |           0.**68** |
| CORE-X-GENRE           | X-GENRE                |       0.35 |       0.41 |              0.53 |           0.3  |
| GINCO-X-GENRE          | CORE-X-GENRE           |       0.37 |       0.45 |              0.31 |           0.68 |
| CORE-X-GENRE           | GINCO-X-GENRE          |       0.28 |       0.35 |              0.53 |           0.24 |
| FTD-X-GENRE            | CORE-X-GENRE           |       0.28 |       0.27 |              0.2  |           0.49 |
| CORE-X-GENRE           | FTD-X-GENRE            |       0.28 |       0.27 |              0.49 |           0.2  |

**MaCoCu-mk**

Results are similar as in MaCoCu-sl. Based on Macro F1, FTD-X-GENE and GINCO-X-GENRE are the most similar to the X-GENRE classifier, with the same Macro F1 scores. Based on Micro F1, GINCO-X-GENRE is more similar. After these pairs, GINCO-X-GENRE and FTD-X-GENRE are similar. CORE-X-GENRE is the least similar to all others.

| Classifier as y_true   | Classifier as y_pred   |   Macro F1 |   Micro F1 |   Macro precision |   Macro recall |
|:-----------------------|:-----------------------|-----------:|-----------:|------------------:|---------------:|
| GINCO-X-GENRE          | X-GENRE                |       **0.63** |       **0.79** |              0.62 |           **0.77** |
| FTD-X-GENRE            | X-GENRE                |       **0.63** |       0.71 |              0.66 |           0.67 |
| X-GENRE                | GINCO-X-GENRE          |       **0.63** |       **0.79** |              **0.77** |           0.62 |
| X-GENRE                | FTD-X-GENRE            |       **0.63** |       0.71 |              0.67 |           0.66 |
| GINCO-X-GENRE          | FTD-X-GENRE            |       0.53 |       0.71 |              0.54 |           0.67 |
| FTD-X-GENRE            | GINCO-X-GENRE          |       0.53 |       0.71 |              0.67 |           0.54 |
| X-GENRE                | CORE-X-GENRE           |       0.51 |       0.62 |              0.44 |           0.71 |
| CORE-X-GENRE           | X-GENRE                |       0.4  |       0.53 |              0.55 |           0.34 |
| GINCO-X-GENRE          | CORE-X-GENRE           |       0.38 |       0.57 |              0.35 |           0.68 |
| FTD-X-GENRE            | CORE-X-GENRE           |       0.34 |       0.44 |              0.27 |           0.5  |
| CORE-X-GENRE           | FTD-X-GENRE            |       0.34 |       0.44 |              0.5  |           0.27 |
| CORE-X-GENRE           | GINCO-X-GENRE          |       0.3  |       0.48 |              0.53 |           0.27 |

Results of comparison based on the apriori rules: see the Mapping to CORE-table (Določanje žanrov > Mapping to CORE), sheet "X-GENRE-schema-apriori-analysis".