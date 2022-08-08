# Comparison of genre datasets: CORE, GINCO and FTD

We compare genre datasets that aim to cover all of the genre diversity found on the web: the CORE dataset, the GINCO dataset and the FTD dataset.

To this end, we perform text classification experiments:
* baseline experiments: in-dataset experiments (training and testing on the same dataset)
* cross-dataset experiments: training on one dataset, applying prediction on the other two - to analyse the comparability of labels (which labels are predicted as which)
* multi-dataset experiments: merging the labels into a joint schema, and training on a combination of all three datasets - using the joint schema, testing on each dataset (+ on a combination of the datasets)
* multi-lingual experiments: extending the multi-dataset experiments by adding the other CORE languages

To simplify experiments, we will perform single-label classification and the texts from CORE and FTD which are labelled with multiple labels will not be used.

We will use the base-sized XLM-RoBERTa model.

The joint schema (merging the FTD labels with the GINCORE labels is based on the [FTD guidelines](https://github.com/ssharoff/genre-keras)):

<img style="width:100%" src="figures/GINCORE-schema-plus-FTD.png">

(FTD categories, marked with a * are not present in the FTD corpus.)

## Repository structure
* artifacts: contains saved models
* data-preparation code: code for preparation of splits for training and testing; some additional data for preparation of splits
* data-sheets-with-all-information: sheets for GINCO, MT-GINCO, CORE and FTD, enriched with all information obtained from the experiments and containing all instances (even those that were not used in the experiments)
* data-splits: datasets, split into train, dev and test splits which were used for classification
* figures: figures with information on results, joint schema etc.
* hyperparameter-search: code with hyperparameter search for FTD - sweep with Wandb (not used in the end)
* results: json and csv files with results of the model training
* root folder: "2.*" - code for training, testing the models and applying prediction to the datasets; "3.*" - code for comparison of labels (based on the prediction on other datasets)

## Table of Contents
* [Experiments overview](#experiments-overview)
* [Information on the datasets](#information-on-the-datasets)
* [Baseline experiments](#baseline-experiments)
* [Comparison of labels based on cross-dataset prediction](#comparison-of-labels-based-on-cross-dataset-prediction)
* [Training on a joint dataset](#training-on-the-joint-schema)
* [X-GENRE: adding X-CORE datasets](#x-genre-adding-x-core-datasets)

## Experiments Overview

As previous experiments have shown that there is little variance between the results, each experiment will be performed once. We will do the following experiments:

1. Baseline experiments (in-dataset experiments) - training and testing on the splits from the same dataset:
    * GINCO-full-set: the GINCO dataset with the full set of GINCO labels, except instances of labels that have less than 10 instances -> 17 labels
    * GINCO-downcast: GINCO dataset a merged set of labels - 9 labels
    * CORE-main: main categories as labels
    * CORE-sub: subcategories as labels

    | Dataset | Micro F1 | Macro F1 |
    |---------|----------|----------|
    | FTD     | 0.739    | 0.74     |
    | GINCO-full-set        |  0.591        | 0.466         |
    | GINCO-downcast        |  0.73        |  0.715        |
    | MT-GINCO-downcast        |   0.72       |  0.723        |
    | CORE-main        |    0.745      |   0.62       |
    | CORE-sub        |    0.661      |   0.394       |

    From the in-dataset experiments, we can see that the FTD dataset obtains the best results, followed by GINCO-downcast and MT-GINCO-downcast, while the GINCO-full-set and CORE-main achieve worse results, especially in terms of macro F1. The CORE-sub shows the problem with a too high granularity of labels, as the macro F1 is low.

    From the results we can see that there is no big difference between the performance of GINCO-downcast (trained on Slovene text) and MT-GINCO-downcast (trained on English text - Slovene text, machine-translated to English).

    For more details, see [Baseline experiments](#baseline-experiments).

2. Applying prediction to other datasets:
    * predict FTD on Sl-GINCO and MT_GINCO
    * predict FTD on CORE
    * predict MT-GINCO (downcast) on FTD and CORE
    * predict SL-GINCO (downcast) on FTD and CORE
    * predict CORE-main on SL-GINCO, MT-GINCO and FTD
    * predict CORE-sub on SL-GINCO, MT-GINCO and FTD

    Comparison between the prediction of FTDs on Slovene and MT text shows that mostly there is not a big difference between prediction on Slovene or English text when the model is trained on English text. Only in 23% instances there is a difference between the FTD labels predicted on SL and MT text. This indicates that prediction of genre seems to be easily cross-lingual. However, it also depends on genres. On some labels, the predictions are worse on MT (Promotion labels), on some it is better (News: 0.24 more correctly predicted instances of News). Regarding the other direction (training on Slovene data, predicting on English), the situation is similar - the GINCO and MT-GINCO predictions on the CORE sample dataset differ only in case of 265 instances (18% of instances) and on the FTD they differ in case of 347 instances (29%).

    Most FTD and GINCO-downcast categories match very well, even when we apply the Slovene classifier to the FTD dataset. The only two FTD categories that are not matched well by the GINCO categories are 'A1 (argumentative) and A17 (review). When we apply the MT-GINCO classifier, the results are better for A12 (promotion) (9 points), A4 (fiction) (26 points), A7 (instruction) (6 points), A8 (news) (7 points), but worse for A14 (academic) (3 points), A16 (information) (9 points), A9 (legal) (13 points).

    The comparison showed that the main CORE categories are not well connected to the FTD categories. The only main CORE category where a majority of instances are identified with a corresponding FTD label, is 'How-To/Instructional' ('A7 (instruction)': 0.713). Some CORE main categories could be described by a combination of FTD categories: 'Interactive Discussion' (forum): 'A1 (argumentative)' + 'A11 (personal)', Opinion': 'A1 (argumentative)' + 'A17 (review)' . Most CORE main labels are predicted with multiple FTD labels where no corresponding label has the majority.

    In contrast to the other direction, FTD categories are identified well with the CORE-main classifier - 7 categories match, while 3 categories are not well predicted. As in the case of the CORE-main predictions to the GINCO dataset, we can see that the category "Opinion" is not matched with the FTD category 'A1 (argumentative)' or 'A11 (personal)' which were expected to be connected. However, here, FTD's Review is better identified as CORE's "Opinion" than in the case of GINCO. As with GINCO, the CORE-main classifier is not capable of recognizing promotion.

   FTD categories match much more with the CORE sub categories than with the main categories. 19 CORE subcategories match very well with FTD categories. Some categories, such as  'Description with Intent to Sell' or 'News Report/Blog' match worse, but they were still predominantly predicted with appropriate FTD category. Around 20 CORE subcategories do not match with FTD categories well, which means that there was no predominantly predicted FTD category which would be appropriate.FTD predictions on CORE sublabels also revealed some issues with the categorization of instances of certain labels ("Magazine Article", "Description of a Thing", "Research Article" etc.). This shows that maybe some of the categories are not to be included in the joint schema which will be used for training a classifier on all of the datasets.

   Suprisingly, the main CORE labels are rather well connected to the GINCO-downcast labels, even when the Slovene classifier is used. The only category that is not connected is the category "Spoken".  Some category are better predicted with MT-GINCO ('How-To/Instructional': 'Instruction' - 4 points better; 'Narrative': 'News/Reporting'- 1 point, 'Opinion': 'Opinion/Argumentation' - 8 points, 'Interactive Discussion': 'Forum' - 10 points; 'Lyrical': 'Other' - 13 points), some are worse ('Informational Persuasion': 'Promotion' - 7 points worse; 'Informational Description/Explanation': 'Information/Explanation' - 5 points worse).

    The CORE-main predictions on Slovene and MT text differ in 182 instances (18%). 12 GINCO categories are relatively well connected to the CORE main categories (with at least on of the classifiers), while the other half (12 categories) are not well connected. Using MT-GINCO improves results in some cases ('News/Reporting': 'Narrative' - 14 points better; 'Opinionated News': 'Narrative' - 13 points better; 'Prose': 'Narrative' - 17 points better), while it gives worse results with some other categories (Forum: 'Interactive Discussion' - 18 points less; Script/Drama - MT-GINCO identifies it as Informational Description/Explanation). The comparison shows that CORE categories and texts are not suited well to be able to recognize some of the genres that are included in the GINCO schema: Correspondence, Promotional categories (Invitation, Promotion, Promotion of a Product, Promotion of Services), List of Summaries/Excerpts. Interestingly, although the CORE includes a category "Opinion" it is not matched well to the GINCO category Opinion/Argumentation, and the GINCO category "Review" which is a CORE subcategory belonging under the main category "Opinion" is not recognized by this main category.

   As with the GINCO labels, the comparison also revealed that FTD labels do not focus on some other labels, that GINCO and CORE define as a separate genre category. For instance, while GINCO and CORE have Forum as a genre category, it is not possible to identify this category with the FTD schema. According to FTD predictions, Forum text are between argumentative and personal texts. This could be a problem if we merge the datasets, because we cannot know how many forum texts are in the FTD dataset, annotated as another category (e.g., as Opinion).

    For more details, see [Comparison of labels based on cross-dataset prediction](#comparison-of-labels-based-on-cross-dataset-prediction)

3. Training on a combination of GINCO + FTD + CORE (joint schema):
    * testing on SL-GINCO and MT-GINCO (joint schema)
    * testing on CORE (joint schema)
    * testing on FTD (joint schema)
    * testing on a combination of GINCO + FTD + CORE (joint schema)
    * testing on EnTenTen (manual analysis whether predicted labels apply)

    For more details, see [Training on a joint dataset](#training-on-the-joint-schema)

6. Multilingual experiments: training on GINCO + FTD + CORE + X-CORE corpora (joint schema):
    * testing on GINCO (GINCO schema)
    * testing on CORE (CORE schema)
    * testing on FTD (FTD schema)
    * (testing on EN-GINCO (GINCO schema))
    * testing on a combination of GINCO + FTD + CORE (joint schema)
    * testing on a combination of all corpora used for training

    For more details, see [X-GENRE: adding X-CORE datasets](#x-genre-adding-x-core-datasets)

## Information on the datasets

Content:
* [Information on CORE](#information-on-core)
* [Information on FTD](#information-on-ftd)
* [Information on GINCO](#information-on-ginco)

### Information on CORE

When preparing the dataset, we:
* discarded instances with no texts (17)
* discarded duplicates (12)

The dataset has 48,420 texts with 459 different main and sub-category label combinations. Regarding main labels, it has 35 different combinations and 297 different sub-category combinations.

Training and testing the model on such a big dataset takes a lot of time and computational sources. In addition to this, a recent article  *Register identification from the unrestricted open Web using the Corpus of Online Registers of English* (Veronika Laippala et al.) showed that the performance of the model does not improve much after being trained on 30% of the data (see figure below). Thus, I decided to use only 30% of the data for baseline experiments and prediction of labels to other datasets. This means that for each CORE-main and CORE-sub, I conducted a stratified split based on the labels and used 30% of the data which I further split into train, test and dev splits.

![](figures/Performance-per-train-data-CORE.png)

#### CORE-main

CORE-main is the CORE dataset, annotated with main categories only (9 categories). For these experiments, we discarded all text that are annotated with more than 1 main category (5686 texts).

| main labels                  | count | percentage |
|-----------------------------------------|-------|------------|
| Narrative                               | 17659 | 41.3231    |
| Informational Description/Explanation | 9314  | 21.7953    |
| Opinion                                 | 7862  | 18.3975    |
| Interactive Discussion                | 3272  | 7.65667    |
| How-To/Instructional                    | 1493  | 3.49371    |
| Informational Persuasion              | 1330  | 3.11228    |
| Lyrical                                 | 639   | 1.4953     |
| Spoken                                  | 583   | 1.36425    |
| Other                                   | 582   | 1.36191    |


|       |   text_length |
|:------|--------------:|
| text count |      42734    |
| mean  |       1236.27 |
| std   |       3167.36 |
| min   |         52    |
| 25%   |        333    |
| 50%   |        630.5  |
| 75%   |       1152    |
| max   |     118278    |

Total number of texts: 42734. For the experiments, I used 40% of the data: 17094 instances. The data is split into 60:20:20 stratified split: train (10256 instance), test and dev split (3419 instances each). The distribution of the labels remained the same:

|                                       |   Count |   Percentage |
|:--------------------------------------|--------:|-------------:|
| Narrative                             |    7064 |     41.3244  |
| Informational Description/Explanation |    3726 |     21.7971  |
| Opinion                               |    3145 |     18.3983  |
| Interactive Discussion                |    1309 |      7.65766 |
| How-To/Instructional                  |     597 |      3.49245 |
| Informational Persuasion              |     532 |      3.1122  |
| Lyrical                               |     255 |      1.49175 |
| Other                                 |     233 |      1.36305 |
| Spoken                                |     233 |      1.36305 |

#### CORE-sub

CORE-sub is the CORE dataset, annotated with subcategories only.

For the experiments, I used roughly 40% of the data: 15,895 instances. Prior to that, I also:
* discarded all texts that are annotated with multiple subcategories (3622)
* discarded all texts that are not annotated with any subcategory (4932)
* discarded instances belonging to categories with less than 10 instances in the subset (15,895 instances): Other Narrative, Other Lyrical, Other How-to, "Other Informational Persuasion", "Advertisement", "Prayer", "Other Spoken", "Other Forum", "Other Opinion", "TV/Movie Script"

The data is split into 60:20:20 stratified split: train (9537 instance), test and dev split (3179 instances each). There are 37 labels. The distribution of the labels remained the same as in the entire dataset:

|                                 |   Count |   Percentage |
|:--------------------------------|--------:|-------------:|
| News Report/Blog                |    4201 |    26.4297   |
| Opinion Blog                    |    1654 |    10.4058   |
| Description of a Thing          |    1403 |     8.82668  |
| Sports Report                   |    1128 |     7.09657  |
| Personal Blog                   |    1108 |     6.97075  |
| Discussion Forum                |     780 |     4.9072   |
| Reviews                         |     721 |     4.53602  |
| Information Blog                |     611 |     3.84398  |
| How-to                          |     527 |     3.31551  |
| Description with Intent to Sell |     437 |     2.74929  |
| Question/Answer Forum           |     421 |     2.64863  |
| Advice                          |     373 |     2.34665  |
| Research Article                |     329 |     2.06983  |
| Description of a Person         |     306 |     1.92513  |
| Religious Blogs/Sermons         |     279 |     1.75527  |
| Song Lyrics                     |     217 |     1.36521  |
| Encyclopedia Article            |     209 |     1.31488  |
| Interview                       |     187 |     1.17647  |
| Historical Article              |     169 |     1.06323  |
| Travel Blog                     |     113 |     0.710915 |
| Short Story                     |     113 |     0.710915 |
| FAQ about Information           |     101 |     0.63542  |
| Legal terms                     |      75 |     0.471846 |
| Recipe                          |      69 |     0.434099 |
| Other Information               |      55 |     0.346021 |
| Persuasive Article or Essay     |      48 |     0.301982 |
| Course Materials                |      47 |     0.29569  |
| Magazine Article                |      29 |     0.182447 |
| Poem                            |      29 |     0.182447 |
| Editorial                       |      26 |     0.163573 |
| Transcript of Video/Audio       |      25 |     0.157282 |
| Reader/Viewer Responses         |      20 |     0.125826 |
| FAQ about How-to                |      19 |     0.119534 |
| Letter to Editor                |      17 |     0.106952 |
| Formal Speech                   |      17 |     0.106952 |
| Technical Report                |      16 |     0.100661 |
| Technical Support               |      16 |     0.100661 |

### Information on FTD

The dataset is available here: https://github.com/ssharoff/genre-keras/blob/master/en.csv

To simplify experiments, we will perform single-label classification. The original dataset allows multiple-label annotation using a scale from 0 to 2 for each label. To simplify experiments, we will regard texts annotated with 2 at a certain category as belonging to this label.

The distribution of the labels:

| Labels         | Count | % (of single labels)|
|--------------------|-------------|--------------|
| **single labels**      | **1547**        |      |
| A1 (argumentative) | 297         | 19,20%       |
| A11 (personal)     | 79          | 5,11%        |
| A12 (promotion)    | 259         | 16,74%       |
| A14 (academic)     | 81          | 5,24%        |
| A16 (information)  | 168         | 10,86%       |
| A17 (review)       | 70          | 4,52%        |
| A22 (non-text)     | 125         | 8,08%        |
| A4 (fiction)       | 94          | 6,08%        |
| A7 (instruction)   | 165         | 10,67%       |
| A8 (news)          | 136         | 8,79%        |
| A9 (legal)         | 73          | 4,72%        |
| **multiple labels**    | **139**         |      |
| **Grand Total**        | **1686**        |              |

Although the FTD article mentions 12 principal categories and 6 optional, it seems that the final FTD dataset is annotated only with 11 categories, out of which one is an "unsuitable" category.

For the experiments, we removed:
* unsuitable texts, marked as "non-text" (125)
* texts, annotated with multiple labels (139)
* duplicated texts (7)

The final number of instances, used for the experiments: 1415.

Dataset that is used for the ML experiments is split into the train-dev-test split according to the label distribution.

|        | train              | test               | dev                |
|:-------|:-------------------|:-------------------|:-------------------|
| count (texts) | 849                | 283                | 283                |

Text length (non-text instances and multiple labels included in the table below):

|       |    length |
|:------|----------:|
| count |   1678    |
| mean  |   1468.09 |
| std   |   4644.93 |
| min   |     31    |
| 25%   |    244    |
| 50%   |    564.5  |
| 75%   |   1291.25 |
| max   | 146922    |

There are 215 texts that are longer than 2000 words, 71 of them are longer than 5000 words and 10 of them are longer than 20,000 words. The analysis shows that the corpus contains very big parts of literary works (e.g., __id__47-FictBalzacH_Goriot_Ia_EN.txt - 22.3k words) and very long UN documents (e.g., __id__214-un - 35.6k words).

### Information on GINCO

We will use paragraphs of texts that are marked as "keep".

Text length:

|       |   text_length |
|:------|--------------:|
| count |      1002     |
| mean  |       362.159 |
| std   |       483.747 |
| min   |        12     |
| 25%   |        98     |
| 50%   |       208     |
| 75%   |       418.75  |
| max   |      4364     |

#### GINCO-full-set

As labels, we used the primary_level_1 labels (the original set without downcasting).Like in experiments with CORE, we discarded instances of categories with less than 10 instances (marked with a * in the table below).

|                            |   Count |   Percentage |
|:---------------------------|--------:|-------------:|
| Information/Explanation    |     130 |  0.129741    |
| News/Reporting             |     115 |  0.11477     |
| Promotion of a Product     |     115 |  0.11477     |
| Opinion/Argumentation      |     114 |  0.113772    |
| List of Summaries/Excerpts |     106 |  0.105788    |
| Opinionated News           |      89 |  0.0888224   |
| Forum                      |      52 |  0.0518962   |
| Instruction                |      38 |  0.0379242   |
| Other                      |      34 |  0.0339321   |
| Invitation                 |      32 |  0.0319361   |
| Promotion of Services      |      32 |  0.0319361   |
| Promotion                  |      30 |  0.0299401   |
| Legal/Regulation           |      17 |  0.0169661   |
| Announcement               |      17 |  0.0169661   |
| Review                     |      17 |  0.0169661   |
| Correspondence             |      16 |  0.0159681   |
| Call                       |      11 |  0.010978    |
| Research Article*           |       9 |  0.00898204  |
| Interview*                  |       8 |  0.00798403  |
| Recipe*                     |       6 |  0.00598802  |
| Prose*                      |       6 |  0.00598802  |
| Lyrical*                    |       4 |  0.00399202  |
| FAQ*                        |       3 |  0.00299401  |
| Script/Drama*               |       1 |  0.000998004 |

The final dataset has 965 texts with 17 different labels. A stratified split was performed in a 60:20:20 manner into a train (579), dev and test spli (each 193). The splits are saved as *data-splits/GINCO-full-set-{train, test, dev}.csv*

The spreadsheet with information on the splits is saved as *data-sheets-with-all-info/GINCO-MT-GINCO-keeptext-with-all-information.csv*.

#### GINCO-downcasted-set

As the results of training the classifier on GINCO-full-set were not great, we will also experiment with a smaller set of labels.

The categories were merged in the following way:
```
{"Script/Drama":"Other", "Lyrical":"Other","FAQ":"Other","Recipe":"Instruction", "Research Article":"Information/Explanation", "Review":"Opinion/Argumentation", "Promotion of Services":"Promotion", "Promotion of a Product":"Promotion", "Invitation":"Promotion", "Correspondence":"Other", "Prose":"Other", "Call":"Other", "Interview":"Other", "Opinionated News":"News/Reporting", "Announcement": "News/Reporting"}
```

The downcasted set (primary_level_4 in the sheet with all information) has 9 labels:

|                            |   Count |   Percentage |
|:---------------------------|--------:|-------------:|
| News/Reporting             |     221 |    0.220559  |
| Promotion                  |     209 |    0.208583  |
| Information/Explanation    |     139 |    0.138723  |
| Opinion/Argumentation      |     131 |    0.130739  |
| List of Summaries/Excerpts |     106 |    0.105788  |
| Other                      |      83 |    0.0828343 |
| Forum                      |      52 |    0.0518962 |
| Instruction                |      44 |    0.0439122 |
| Legal/Regulation           |      17 |    0.0169661 |

The final dataset has 1002 instances, split into 60:20:20 stratified split - into train (601 instances), dev (201 instances) and test (200 instances) files.

The splits are saved as *data-splits/GINCO-downcast-{train, test, dev}.csv*

## Baseline experiments

### FTD
I used the wandb library to evaluate the optimal number of epochs by performing evaluation during training. By analysing the training and evaluation loss, I opted for the epoch number = 10.

Code for training: *2.1-FTD-classifier-training-and-saving.ipynb*

The hyperparameters that I used:

```
            "overwrite_output_dir": True,
            "num_train_epochs": 10,
            "train_batch_size":8,
            "learning_rate": 1e-5,
            # Use these parameters if you want to evaluate during training
            #"evaluate_during_training": True,
            #"evaluate_during_training_steps": steps_per_epoch*10,
            #"evaluate_during_training_verbose": True,
            #"use_cached_eval_features": True,
            #'reprocess_input_data': True,
            "labels_list": LABELS,
            "max_seq_length": 512,
            "save_steps": -1,
            # Only the trained model will be saved - to prevent filling all of the space
            "save_model_every_epoch":False,
            "wandb_project": 'FTD-learning-manual-hyperparameter-search',
            "silent": True,
```

The trained model was saved to the Wandb repository and can be accessed for testing (see code *2.2-FTD-classifier-testing-and-applying-on-other-datasets.ipynb*).

Load the FTD model from Wandb:
```
artifact = run.use_artifact('tajak/FTD-learning-manual-hyperparameter-search/FTD-classifier:v1', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

The results on dev file: Macro f1: 0.759, Micro f1: 0.749

The results on test file: Macro f1: 0.74, Micro f1: 0.739

![Confusion matrix for training and testing on FTD](figures/CM-FTD-classifier-on-FTD-test.png)

After the model was created, I applied it to the entire GINCO - to Slovene text (FTD_pred_on_SL) and to MT text (FTD_pred_on_MT) and the test split of the CORE dataset (because it would take multiple days to apply predictions on the whole dataset) and added FTD predictions to them, so that we will be able to analyze how the labels overlap. Prediction takes 20 minutes for 1,000 instances.

The datasets with FTD predictions:
- FTD dev and test split: *results/testing-FTD-model-on-dev-sheet-with-predictions.csv, **results/FTD-classifier-predictions-on-test-sheet-with-predictions.csv*; 
- the GINCO dataset with FTD predictions: *data-sheets-with-all-info/GINCO-MT-GINCO-keeptext-split-file-with-all-information.csv*;
- the CORE dataset with FTD predictions: *data-sheets-with-all-info/CORE-all-information.csv*

### GINCO-full-set

I evaluated the model during training to search for the optimum epoch number. As can it be seen from the figure below, it was between 12 and 20 (the global steps needs to be divided by 72 to get the epoch number), since afterwards the eval_loss starts rising again.

![Evaluation during training to find the optimum number of epochs](figures/GINCO-full-set-epoch-number-search.png)

Then I trained the model for 12, 15, 20 and 25 epochs (results in *results/GINCO-Experiments-Results.json*), evaluating it on dev split and the results revealed that the optimum number of epochs is 20.

Final hyperparameters:
```
        args= {
            "overwrite_output_dir": True,
            "num_train_epochs": 20,
            "train_batch_size":8,
            "learning_rate": 1e-5,
            "labels_list": LABELS,
            "max_seq_length": 512,
            "save_steps": -1,
            # Only the trained model will be saved - to prevent filling all of the space
            "save_model_every_epoch":False,
            "wandb_project": 'GINCO-hyperparameter-search',
            "silent": True,
            }

```

To load the GINCO-full-set model from Wandb:
```
artifact = run.use_artifact('tajak/GINCO-hyperparameter-search/GINCO-full-set-classifier:v0', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

The results on dev file: Macro f1: 0.539, Micro f1: 0.668

The results on test file: Macro f1: 0.466, Micro f1: 0.591

Confusion matrix on the test split:
![Confusion matrix for the full set on the test split](figures/CM-GINCO-full-set-classifier-on-test.png)

### GINCO-downcast

I evaluated the model during training to search for the optimum epoch number. As can it be seen from the figure below, it was between 8 and 15 (the global steps needs to be divided by 75 to get the epoch number), since afterwards the eval_loss starts rising again.

![Evaluation during training to find the optimum number of epochs](figures/GINCO-downcast-epoch-search.png)

Then I trained the model for 8, 10, 15, 20, 25 and 30 epochs (results in *results/GINCO-Experiments-Results.json*), evaluating it on dev split and the results revealed that the optimum number of epochs is 15.

Final hyperparameters:
```
        args= {
            "overwrite_output_dir": True,
            "num_train_epochs": 15,
            "train_batch_size":8,
            "learning_rate": 1e-5,
            "labels_list": LABELS,
            "max_seq_length": 512,
            "save_steps": -1,
            # Only the trained model will be saved - to prevent filling all of the space
            "save_model_every_epoch":False,
            "wandb_project": 'GINCO-hyperparameter-search',
            "silent": True,
            }
```

To load the GINCO-downcast model from Wandb:
```
import wandb
run = wandb.init()
artifact = run.use_artifact('tajak/GINCO-hyperparameter-search/GINCO-downcast-classifier:v0', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

The results on dev file: Macro f1: 0.741, Micro f1: 0.701

The results on test file: Macro f1: 0.73, Micro f1: 0.715

![](figures/CM-GINCO-downcast-on-test.png)

As can be seen on the confusion matrix, the model is even able to capture List of Summaries :D

### MT-GINCO-downcast

As the GINCO-downcast had good results, I also trained a model on MT-GINCO with downcast labels to be able to compare their performance and see if there is any difference between the Slovene classifier and MT classifier when applied on other datasets.

I used the same hyperparameters and the instances are in the same splits as in GINCO-downcast experiments, the only difference is that the text is in English.

To load the MT-GINCO-downcast model from Wandb:
```
import wandb
run = wandb.init()
artifact = run.use_artifact('tajak/GINCO-hyperparameter-search/MT-GINCO-downcast-classifier:v0', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

The results on dev file: Macro f1: 0.806, Micro f1: 0.781
The results on test file: Macro f1: 0.723, Micro f1: 0.72

![](figures/CM-MT-GINCO-downcast-on-test.png)

### CORE-main

I evaluated the model during training to search for the optimum epoch number which revealed the optimum number of epochs to be between 2 and 6 epochs. Then I trained the model for 2, 4, and 6 epochs and the optimum number of epochs revealed to be 4.

Final hyperparameters:
```
        args= {
            "overwrite_output_dir": True,
            "num_train_epochs": 4,
            "train_batch_size":8,
            "learning_rate": 1e-5,
            "labels_list": LABELS,
            "max_seq_length": 512,
            "save_steps": -1,
            # Only the trained model will be saved - to prevent filling all of the space
            "save_model_every_epoch":False,
            "wandb_project": 'CORE-hyperparameter-search',
            "silent": True,
            }
```

To load the CORE-main model from Wandb:
```
import wandb
run = wandb.init()
artifact = run.use_artifact('tajak/CORE-hyperparameter-search/CORE-main-classifier:v0', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

Results on the dev file: Macro f1: 0.623, Micro f1: 0.736
Results on test file: Macro f1: 0.62, Micro f1: 0.745

![](figures/CM-CORE-main-on-test.png)

### CORE-sub

I evaluated the model during training to search for the optimum epoch number which revealed the optimum number of epochs to be between 2 and 8 epochs. Then I trained the model for 2, 4, 6 and 8 epochs and the optimum number of epochs revealed to be 6.

Final hyperparameters:
```
        args= {
            "overwrite_output_dir": True,
            "num_train_epochs": 6,
            "train_batch_size":8,
            "learning_rate": 1e-5,
            "labels_list": LABELS,
            "max_seq_length": 512,
            "save_steps": -1,
            # Only the trained model will be saved - to prevent filling all of the space
            "save_model_every_epoch":False,
            "wandb_project": 'CORE-hyperparameter-search',
            "silent": True,
            }
```

To load the CORE-sub model from Wandb:
```
import wandb
run = wandb.init()
artifact = run.use_artifact('tajak/CORE-hyperparameter-search/CORE-sub-classifier:v0', type='model')
artifact_dir = artifact.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", artifact_dir)
```

Results on the dev file: Macro f1: 0.396, Micro f1: 0.662
Results on test file: Macro f1: 0.394, Micro f1: 0.661

![](figures/CM-CORE-sub-on-test.png)

## Comparison of labels based on cross-dataset prediction

### FTD and GINCO / GINCO and FTD

#### FTD to GINCO
I applied the FTD classifier in separate prediction runs once to Slovene text and once to the text that was machine-translated to English. The code with comparison and results is here: *3.1-Compare-FTD-labels-on-GINCO.ipynb*

Main conclusions:

Comparison between the prediction on Slovene and MT text shows that mostly there is not a big difference between prediction on Slovene or English text. Only in 227 instances (out of 1002 - 23%) there is a difference between the FTD labels predicted on SL and MT text. This indicates that prediction of genre seems to be easily cross-lingual. However, it also depends on genres. On some labels, the predictions are worse on MT (Promotion labels, Prose), on some it is better (Instruction - 0.05 more correctly predicted instances, Research Article - 0.10 more predicted instances, Information/Explanation). On MT, there is a much better identification of News (0.24 more correctly predicted instances) and Opinionated News (0.10 more correctly predicted instances). In other cases, there is no difference or just a slight difference (e.g., Legal/Regulation).

Labels that were very well predicted on Slovene text: Instruction ('A7 (instruction)': 0.79 - precentage of all instruction instances), Invitation ('A12 (promotion)': 0.875), Legal/Regulation ('A9 (legal)': 1.0), Promotion of Services ('A12 (promotion)': 0.94), Promotion of a Product ('A12 (promotion)': 0.85, Promotion ('A12 (promotion)': 0.8), Prose ('A4 (fiction)': 1.0), Recipe ('A7 (instruction)': 0.83), Research Article ('A14 (academic)': 0.67), Review ('A17 (review)': 0.65), Information/Explanation ('A16 (information)': 0.6)

The comparison of labels:

1. Very well connected labels (similar):
    - Instruction: 'A7 (instruction)'
    - Invitation: 'A12 (promotion)'
    - Legal/Regulation: 'A9 (legal)'
    - Promotion of Services: 'A12 (promotion)'
    - Promotion of a Product: 'A12 (promotion)'
    - Promotion: 'A12 (promotion)'
    - Prose: 'A4 (fiction)'
    - Recipe: 'A7 (instruction)'
    - Research Article: 'A14 (academic)'
    - Review: 'A17 (review)'
    - Information/Explanation: 'A16 (information)'


2. A bit less connected labels (lower percentage of GINCO instances with the "correct" FTD label):
    - News/Reporting: 'A8 (news)' - very well connected in MT-GINCO, less connected on SL-GINCO
    - Opinion/Argumentation: 'A1 (argumentative) (0.39 - GINCO, 0.46 - MT-GINCO) + 'A11 (personal)' (0.14 - GINCO, 0.18 - MT-GINCO)
    - Opinionated News: 0.50-0.60 identified as 'A8 (news)', 20% as 'A1 (argumentative)' (on MT-GINCO)

3. Some GINCO labels were predicted by a variety of FTD labels - there is no majority FTD label:
    - Announcement: the closest to 'A12 (promotion)': 0.47 on SL text; on MT text, the main labels are 'A12 (promotion)': 0.41 and 'A8 (news)': 0.41 
    - Call: on SL mostly connected to 'A12 (promotion)': 0.73; on MT text, it is divided between 'A1 (argumentative)', 'A12 (promotion)' and 'A9 (legal)' (0.27 each)
    - Correspondence: the closest to 'A1 (argumentative)': 0.375 on SL, on MT similar, but there is also a lot of 'A7 (instruction)': 0.25
    - FAQ: 'A12 (promotion)': 0.67 in SL,  A7 (instruction)': 0.66 and 33 % as 'A12 (promotion)' on MT - to note: there are only 3 FAQ instances
    - Forum: mostly 'A1 (argumentative)' and 'A11 (personal)'; easier to identify as these two on MT than SL
    - Interview: 'A12 (promotion)': 0.375 , 'A17 (review)': 0.25 on SL;  'A1 (argumentative)': 0.25, 'A11 (personal)': 0.375, 'A12 (promotion)': 0.25, and 'A17 (review)': 0.125 on MT.
    - List of Summaries/Excerpts: most vague category - identified as all of the FTD categories.
    - Lyrical: 'A11 (personal)' or 'A4 (fiction)' (note: there are only 4 instances)
    - Other: predicted with various FTD labels, mostly 'A12 (promotion)': 0.44 (less Promotion on MT)
    - Script/Drama (1): not well predicted - as 'A16 (information)' - but there is only one instance.

#### GINCO/MT-GINCO to FTD

Then I applied GINCO-downcast and MT-GINCO downcast classifiers to the FTD dataset. Also in this direction (trained on Slovene/English data, predicted on English data), it seems that there is not a big difference between cross-lingual and monolingual prediction. The GINCO and MT-GINCO predictions differ only in case of 347 instances (29%).

What we can see based on the GINCO predictions on the FTD labels, is (first, the information for prediction of GINCO-downcast is given, followed by the information what is different on predictions by the MT-GINCO-downcast model):

Most FTD and GINCO-downcast categories match very well, even when we apply the Slovene classifier to the FTD dataset. The only two FTD categories that are not matched well by the GINCO categories are 'A1 (argumentative) and A17 (review). When we apply the MT-GINCO classifier, the results are better for A12 (promotion) (9 points), A4 (fiction) (26 points), A7 (instruction) (6 points), A8 (news) (7 points), but worse for A14 (academic) (3 points), A16 (information) (9 points), A9 (legal) (13 points).

1. Categories that match well:
* 'A11 (personal)': 'Opinion/Argumentation': 0.696
* 'A12 (promotion)': 'Promotion': 0.593; with MT-GINCO better: 'Promotion': 0.683
* 'A14 (academic)': 'Information/Explanation': 0.81; with MT-GINCO slightly worse: 'Information/Explanation': 0.78
* 'A16 (information)': 'Information/Explanation': 0.815; with MT-GINCO slightly worse: 'Information/Explanation': 0.73
* 'A4 (fiction)': 'Other': 0.54; with MT-GINCO much better: 'Other': 0.808
* 'A7 (instruction)': 'Instruction': 0.61, 'Instruction': 0.66
* 'A8 (news)': 'News/Reporting': 0.74, 'News/Reporting': 0.81
* 'A9 (legal)': Legal/Regulation': 0.64, 'Legal/Regulation': 0.51

2. FTD categories that were not identified well:
* 'A1 (argumentative)':  'Information/Explanation': 0.263, 'News/Reporting': 0.246, 'Opinion/Argumentation': 0.229; with MT-GINCO only slightly better: 'Opinion/Argumentation': 0.28,  less Information/Explanation
* 'A17 (review)': 'Promotion': 0.29, 'Information/Explanation': 0.19; with MT-GINCO other categories, but not better: 'Promotion': 0.32, 'Opinion/Argumentation': 0.29

### FTD and CORE-main categories

#### FTD to CORE-main

The comparison showed that the main CORE categories are not well predicted with the FTD categories. The only main CORE category where a majority of instances are identified with a corresponding FTD label, is 'How-To/Instructional' ('A7 (instruction)': 0.713). Some CORE main categories could be described by a combination of FTD categories: 'Interactive Discussion' (forum): 'A1 (argumentative)' + 'A11 (personal)', Opinion': 'A1 (argumentative)' + 'A17 (review)' . Most CORE main labels are predicted with multiple FTD labels where no corresponding label has the majority.

Comparison of main CORE labels and FTD labels:

1. Well-connected:
* 'How-To/Instructional': 'A7 (instruction)': 0.713 (percentage of instances of Interactive Discussion, identified as A1)

2. Not well connected (no clear majority label/majority label does not seem to be appropriate):
* 'Interactive Discussion': mostly 'A1 (argumentative)': 0.315 + 'A11 (personal)': 0.289,  'A7 (instruction)': 0.239
* 'Narrative': 'A8 (news)': 0.48, 'A1 (argumentative)': 0.228
* 'Opinion': 'A1 (argumentative)': 0.467, 'A17 (review)': 0.230
* 'Informational Description/Explanation': A12 (promotion)': 0.228, 'A16 (information)': 0.21, 'A1 (argumentative)': 0.189
* 'Lyrical': 'A11 (personal)': 0.577
* 'Informational Persuasion': 'A12 (promotion)': 0.40, 'A1 (argumentative)': 0.185, 'A17 (review)': 0.258
* 'Spoken': 'A1 (argumentative)': 0.30, 'A11 (personal)': 0.278, 'A17 (review)': 0.252

As with the GINCO labels, the comparison also revealed that FTD labels do not focus on some other labels, that GINCO and CORE define as a separate genre category. For instance, while GINCO and CORE have Forum as a genre category, it is not possible to identify this category with the FTD schema. According to FTD predictions, Forum text are between argumentative and personal texts. This could be a problem if we merge the datasets, because we cannot know how many forum texts are in the FTD dataset, annotated as another category (e.g., as Opinion).

#### CORE-main to FTD

In contrast to the other direction, FTD categories are identified well with the CORE-main classifier - 7 categories match, while 3 categories are not well predicted. As in the case of the CORE-main predictions to the GINCO dataset, we can see that the category "Opinion" is not matched with the FTD category 'A1 (argumentative)' or 'A11 (personal)' which were expected to be connected. However, here, FTD's Review is better identified as CORE's "Opinion" than in the case of GINCO. As with GINCO, the CORE-main classifier is not capable of recognizing promotion.

1. Categories that are connected well:
* 'A11 (personal)': 'Narrative': 0.48
* A14 (academic): Informational Description/Explanation': 0.91
* 'A16 (information)': 'Informational Description/Explanation': 0.91
* 'A17 (review)': 'Opinion': 0.56
* 'A4 (fiction)': 'Narrative': 0.80
* 'A8 (news)': 'Narrative': 0.74
* 'A9 (legal)': 'Informational Description/Explanation': 0.99

2. Categories that are not connected well/wrongly predicted:
* 'A1 (argumentative)': 'Informational Description/Explanation': 0.51,  'Opinion': 0.22
* 'A12 (promotion)': 'Informational Description/Explanation': 0.69
* 'A7 (instruction)': 'How-To/Instructional': 0.44, 'Informational Description/Explanation': 0.42

### FTD and CORE sub categories

Predictions of FTD categories to the CORE instances shows that they match much more with the CORE sub categories than with the main categories. 19 CORE subcategories match very well with FTD categories. Some categories, such as  'Description with Intent to Sell' or 'News Report/Blog' match worse, but they were still predominantly predicted with appropriate FTD category. Around 20 CORE subcategories do not match with FTD categories well, which means that there was no predominantly predicted FTD category which would be appropriate.

FTD predictions on CORE sublabels possibly also revealed some issues with the categorization of instances of certain labels. For instance:
* 'Description of a Thing' which is a category that belongs under 'Informational Description/Explanation' main category, was mainly identified as 'A12 (promotion)': 0.37*
* "Historical Article" and "Magazine Article" which belong under "Narrative" main category are identified as having a lot of argumentative properties based on FTD ('A1 (argumentative)')
* "Informational Blog" and "Research Article" which belong under 'Informational Description/Explanation' were also predominantly identified as 'A1 (argumentative)' text
* "Prayer" is mostly identified as 'A16 (information)'

This shows that maybe some of the categories are not to be included in the joint schema which will be used for training a classifier on all of the datasets.

1. Categories that match well:
* 'Advertisement': 'A12 (promotion)': 0.67
* 'Editorial': 'A1 (argumentative)': 0.92
* 'Encyclopedia Article': 'A16 (information)': 0.82
* 'FAQ about How-to':  'A7 (instruction)': 0.66
* 'How-to':  'A7 (instruction)': 0.74
* 'Formal Speech': 'A1 (argumentative)': 0.78
* 'Legal terms': 'A9 (legal)': 0.789
* 'Letter to Editor': 'A1 (argumentative)': 0.875
* 'Opinion Blog': 'A1 (argumentative)': 0.66
* 'Personal Blog: 'A11 (personal)': 0.677
* 'Persuasive Article or Essay': 'A1 (argumentative)': 0.75
* 'Religious Blogs/Sermons': 'A1 (argumentative)': 0.67
* 'Reviews': 'A17 (review)': 0.72
* 'Short Story': 'A4 (fiction)': 0.80
* 'Sports Report': A8 (news)': 0.759
* 'Technical Support': 'A7 (instruction)': 0.875
* Other Informational Persuasion': 'A1 (argumentative)': 1.0
* 'Other Opinion': 'A1 (argumentative)': 0.67
* 'Other Spoken': 'A11 (personal)': 0.67


2. Categories that match, but less well:
* 'Advice': 'A7 (instruction)': 0.51
* 'Description with Intent to Sell': A12 (promotion)': 0.465, 'A17 (review)': 0.283
* 'News Report/Blog': 'A8 (news)': 0.55, 'A1 (argumentative)': 0.28
* 'Recipe': 'A7 (instruction)': 0.47, 'A11 (personal)': 0.35
* 'Travel Blog': 'A11 (personal)': 0.438, A17 (review)': 0.25, 'A12 (promotion)': 0.19

3. CORE sub categories with no (appropriate) majority FTD label:
* 'Course Materials': 'A7 (instruction)': 0.30 or 'A16 (information)': 0.30
* 'Description of a Person':  'A16 (information)': 0.33, 'A17 (review)': 0.228
* 'Description of a Thing': 'A12 (promotion)': 0.37
* 'Discussion Forum': 'A1 (argumentative)': 0.37, 'A11 (personal)': 0.32
* 'FAQ about Information': 'A7 (instruction)': 0.44, 'A12 (promotion)': 0.28
* 'Historical Article': 'A16 (information)': 0.423, 'A1 (argumentative)': 0.36
* 'Information Blog': 'A1 (argumentative)': 0.29, 'A7 (instruction)': 0.21
* 'Interview': ''A17 (review)': 0.36, A1 (argumentative)': 0.18, 'A11 (personal)': 0.265
* 'Magazine Article': 'A1 (argumentative)': 0.47
* 'Poem': 'A4 (fiction)': 0.33, A11 (personal)': 0.2, 'A17 (review)': 0.27
* 'Prayer': A16 (information)': 0.67
* 'Question/Answer Forum':  'A7 (instruction)': 0.357, 'A11 (personal)': 0.23, A1 (argumentative)': 0.209
* 'Reader/Viewer Responses': 'A17 (review)': 0.3, 'A12 (promotion)': 0.3, 'A7 (instruction)': 0.2
* 'Research Article': 'A1 (argumentative)': 0.39, 'A14 (academic)': 0.376
* 'Song Lyrics': 'A11 (personal)': 0.63 (matches well, but debatable if it's appropriate category)
* 'TV/Movie Script': A11 (personal)': 0.6
* 'Technical Report': 'A1 (argumentative)': 0.375, 'A7 (instruction)': 0.25 
* 'Transcript of Video/Audio': 'A1 (argumentative)': 0.83
* 'Other Information':  'A7 (instruction)': 0.30, 'A16 (information)': 0.185

### GINCO/MT-GINCO and CORE labels

The analysis showed that the predictions of GINCO and MT-GINCO on CORE texts are mostly the same - the GINCO and MT-GINCO predictions differ only in case of 265 instances (18% of instances).

#### GINCO/MT-GINCO to CORE-main

Suprisingly, the main CORE labels are rather well connected to the GINCO-downcast labels, even when the Slovene classifier is used. The only category that is not connected is the category "Spoken".  Some category are better predicted with MT-GINCO ('How-To/Instructional': 'Instruction' - 4 points better; 'Narrative': 'News/Reporting'- 1 point, 'Opinion': 'Opinion/Argumentation' - 8 points, 'Interactive Discussion': 'Forum' - 10 points; 'Lyrical': 'Other' - 13 points), some are worse ('Informational Persuasion': 'Promotion' - 7 points worse; 'Informational Description/Explanation': 'Information/Explanation' - 5 points worse).

1. Well-connected:
* 'How-To/Instructional': 'Instruction': 0.698; with MT-GINCO better: 'Instruction': 0.736
* 'Informational Persuasion': 'Promotion': 0.59; with MT-GINCO worse: 'Promotion': 0.52
* 'Narrative': 'News/Reporting': 0.605; with MT-GINCO slightly better: 'News/Reporting': 0.616
* 'Opinion': 'Opinion/Argumentation': 0.548; with MT-GINCO better: 'Opinion/Argumentation': 0.629

2. A bit less connected:
* 'Informational Description/Explanation': 'Information/Explanation': 0.433; with MT-GINCO worse: 'Information/Explanation': 0.38, 'Promotion': 0.19
* 'Interactive Discussion': 'Forum': 0.495, 'Opinion/Argumentation': 0.22; with MT-GINCO better: 'Forum': 0.60
* 'Lyrical': 'Other': 0.478, 'Opinion/Argumentation': 0.21; with MT-GINCO better: 'Other': 0.61

2. Not well connected (no clear majority label/majority label does not seem to be appropriate):
* 'Spoken': 'News/Reporting': 0.304, 'Opinion/Argumentation': 0.304, 'Other': 0.173; with MT-GINCO similar: 'News/Reporting': 0.30, 'Opinion/Argumentation': 0.43, 'Other': 0.21

#### CORE-main to GINCO/MT-GINCO

The CORE-main predictions on Slovene and MT text differ in 182 instances (18%). 12 GINCO categories are relatively well connected to the CORE main categories (with at least on of the classifiers), while the other half (12 categories) are not well connected. Using MT-GINCO improves results in some cases ('News/Reporting': 'Narrative' - 14 points better; 'Opinionated News': 'Narrative' - 13 points better; 'Prose': 'Narrative' - 17 points better), while it gives worse results with some other categories (Forum: 'Interactive Discussion' - 18 points less; Script/Drama - MT-GINCO identifies it as Informational Description/Explanation). The comparison shows that CORE categories and texts are not suited well to be able to recognize some of the genres that are included in the GINCO schema: Correspondence, Promotional categories (Invitation, Promotion, Promotion of a Product, Promotion of Services), List of Summaries/Excerpts. Interestingly, although the CORE includes a category "Opinion" it is not matched well to the GINCO category Opinion/Argumentation, and the GINCO category "Review" which is a CORE subcategory belonging under the main category "Opinion" is not recognized by this main category.

If there is no information regarding the MT-GINCO results, they are the same as the GINCO.

1. Well connected labels:
* FAQ: 'Informational Description/Explanation': 0.67
* Forum: 'Interactive Discussion': 0.76; much worse with MT-GINCO: 'Interactive Discussion': 0.58, 'Narrative': 0.15
* 'Information/Explanation': 'Informational Description/Explanation': 0.907
* Interview: 'Spoken': 0.875
* 'Script/Drama': 'Spoken': 1.0; much worse with MT-GINCO: 'Informational Description/Explanation': 1.0
* 'Legal/Regulation': 'Informational Description/Explanation': 1.0
* Lyrical': 'Lyrical': 0.75
* 'News/Reporting': between 'Informational Description/Explanation': 0.496 and 'Narrative': 0.452; better with MT-GINCO: 'Narrative': 0.59, 'Informational Description/Explanation': 0.37
* 'Opinionated News': between 'Narrative': 0.52 and Informational Description/Explanation': 0.348; better results with MT-GINCO: 'Narrative': 0.65
* 'Prose': 'Narrative': 0.5, even better with MT-GINCO: 'Narrative': 0.67
* 'Recipe': 'How-To/Instructional': 1.0
* 'Research Article': 'Informational Description/Explanation': 1.0

2. Not well connected: 
* Announcement: 'Informational Description/Explanation': 0.94
* Call: 'Informational Description/Explanation': 1.0
* 'Correspondence': 'Interactive Discussion': 0.5; with MT-GINCO between 'Informational Description/Explanation': 0.31 and 'Interactive Discussion': 0.31
* 'Instruction': split between 'How-To/Instructional': 0.42 and 'Informational Description/Explanation': 0.53; with MT-GINCO a bit more 'How-To/Instructional': 0.5
* 'Invitation': 'Informational Description/Explanation': 0.93; even more Informational Description/Explanation with MT-GINCO:  'Informational Description/Explanation': 0.97
* 'List of Summaries/Excerpts': various categories; with MT-GINCO mostly  'Informational Description/Explanation': 0.42, 'Narrative': 0.30
* 'Opinion/Argumentation': 'Informational Description/Explanation': 0.438, 'Spoken': 0.20; even more scattered across categories with MT-GINCO: 'Informational Description/Explanation': 0.33l, 'Informational Description/Explanation': 0.33, 'Spoken': 0.22
* Other: mostly 'Informational Description/Explanation': 0.56; similar with MT-GINCO: 'Informational Description/Explanation': 0.47
* 'Promotion': 'Informational Description/Explanation': 0.7
* 'Promotion of Services':  'Informational Description/Explanation': 0.84
* 'Promotion of a Product': 'Informational Description/Explanation': 0.49; a bit "better" with MT-GINCO: 'Informational Description/Explanation': 0.51, but also 'Informational Persuasion': 0.4
*  'Review': 'Informational Description/Explanation': 0.294, various other categories; even worse with MT-GINCO: 'Spoken': 0.41


### GINCO/MT-GINCO to CORE-sub

If we compare CORE subcategories and GINCO-downcast categories based on the GINCO predictions, we see that 17 CORE subcategories match very well with GINCO categories, 7 match, but less well, and 19 categories do not match well. With some categories, the prediction of MT classifier is better ('Discussion Forum': 'Forum' - 9 points better; 'How-to: 'Instruction' - 4 point better; 'Opinion Blog': 'Opinion/Argumentation' and 'Personal Blog': 'Opinion/Argumentation' - 2 points better; 'Sports Report': 'News/Reporting' - 7 points better; 'Reviews': 'Opinion/Argumentation': 15 points better; 'Song Lyrics': 'Other' - 15 points), in some worse ('Description with Intent to Sell': 'Promotion' - 6 points worse; 'Encyclopedia Article': 'Information/Explanation' - 5 points; 'Historical Article': 'Information/Explanation' - 26 points; 'Persuasive Article or Essay': 'Opinion/Argumentation' - 40 points worse; 'Recipe': 'Instruction' - 16 points worse; 'Travel Blog': 'Opinion/Argumentation' - 19 points worse; 'Legal terms': 'Legal/Regulation': 0.429 on SL, no Legal/Regulation on MT). 

1. Categories that match well:
* 'Advertisement' (1 instance): 'Promotion': 1.0
* 'Course Materials': 'Information/Explanation': 0.75
* 'Description with Intent to Sell': 'Promotion': 0.658; slightly worse with MT: 'Promotion': 0.60
* 'Discussion Forum': 'Forum': 0.575; better with MT: 'Forum': 0.67
* 'Encyclopedia Article': 'Information/Explanation': 0.75; slightly worse with MT: 'Information/Explanation': 0.7
* 'Historical Article': 'Information/Explanation': 0.875; worse with MT:   'Information/Explanation': 0.62, 'Opinion/Argumentation': 0.19
*  'How-to: 'Instruction': 0.63; better with MT: 'Instruction': 0.71
* 'News Report/Blog': 'News/Reporting': 0.79; the same with MT
* 'Opinion Blog': 'Opinion/Argumentation': 0.67; slightly better with MT: 'Opinion/Argumentation': 0.69
* 'Personal Blog': 'Opinion/Argumentation': 0.798; better with MT: 'Opinion/Argumentation': 0.82
* 'Persuasive Article or Essay': 'Opinion/Argumentation': 0.6; much worse with MT: 'Opinion/Argumentation': 0.2
* 'Recipe': 'Instruction': 0.83; worse with MT: 'Instruction': 0.67
* 'Research Article': 'Information/Explanation': 0.677; the same with MT
* 'Sports Report': 'News/Reporting': 0.67; better with MT: 'News/Reporting': 0.74
* 'Technical Report' (1 instance): 'Information/Explanation': 1.0; the same with MT
* 'Technical Support' (1 instance): 'Instruction': 1.0; the same with MT
* 'Travel Blog': 'Opinion/Argumentation': 0.636; worse with MT: 'Opinion/Argumentation': 0.45,'Promotion': 0.27
* 'Formal Speech' (2 instances): 'Opinion/Argumentation': 0.5, 'Other': 0.5; MT:  'Opinion/Argumentation': 1.0
* 'Letter to Editor' (1 instance): 'Forum': 1.0; much better with MT: 'Opinion/Argumentation': 1.0

2. Categories that match, but less well:
* 'Advice': 'Opinion/Argumentation': 0.34, 'Instruction': 0.285; slightly better with MT: 'Instruction': 0.4, 'Opinion/Argumentation': 0.31
* 'Description of a Person': 'Information/Explanation': 0.448, 'Opinion/Argumentation': 0.206; slightly more Opinion with MT
* 'FAQ about How-to': 'Instruction': 0.5, 'Promotion': 0.5; the same with MT
* 'FAQ about Information': 'Information/Explanation': 0.56, 'Instruction': 0.22, 'Promotion': 0.22; the same with MT
* 'Legal terms': 'Legal/Regulation': 0.429, 'Instruction': 0.29; much worse with MT: 'Instruction': 0.43, Information/Explanation': 0.43; 'Legal/Regulation': 0.0
* 'Reviews': 'Opinion/Argumentation': 0.5, 'Promotion': 0.279; much better with MT: 'Opinion/Argumentation': 0.647
* 'Song Lyrics': 'Other': 0.5, 'Opinion/Argumentation': 0.2; better with MT: 'Other': 0.65
* 'Question/Answer Forum': 'Forum': 0.35, 'Other': 0.2; better with MT: 'Forum': 0.475, 'Instruction': 0.15

3. CORE sub categories with no (appropriate) majority GINCO-downcast label:
* 'Description of a Thing': 'Information/Explanation': 0.34, 'Promotion': 0.288; more Promotion with MT: 'Promotion': 0.31, 'Information/Explanation': 0.27
* 'Editorial' (2 instances): 'News/Reporting': 1.0; the same with MT
* 'Information Blog': 'Information/Explanation': 0.33, 'News/Reporting': 0.19; worse with MT: 'Information/Explanation': 0.26, 'Opinion/Argumentation': 0.21
* 'Interview': 'News/Reporting': 0.44, 'Opinion/Argumentation': 0.22; similar with MT, except more Opinion/Argumentation
* 'Magazine Article' (3 instances): 'Information/Explanation': 0.33, 'List of Summaries/Excerpts': 0.33, 'News/Reporting': 0.33; different with MT, but not better: 'Opinion/Argumentation': 0.67
* 'Other Forum' (1 instance):  'Opinion/Argumentation': 1.0; the same with MT
* 'Other Information':  'Promotion': 0.4, 'Information/Explanation': 0.2; the same with MT
* 'Other Opinion' (1 instance) 'Promotion': 1.0; the same with MT
* 'Other Spoken' (1 instance): 'Opinion/Argumentation': 1.0; the same with MT
* 'Poem' (3 instances): 'Information/Explanation': 0.33, 'Opinion/Argumentation': 0.33, 'Other': 0.33; same with MT
* 'Prayer' (1 instance): 'Information/Explanation': 1.0; same with MT   
* 'Reader/Viewer Responses' (2 instances): 'Opinion/Argumentation': 0.5, 'Other': 0.5; better with MT: 'Forum': 0.5, 'Opinion/Argumentation': 0.5
* 'Religious Blogs/Sermons': 'Opinion/Argumentation': 0.30, 'Information/Explanation': 0.27; better (?) with MT: 'Opinion/Argumentation': 0.615
* 'Short Story': 'Opinion/Argumentation': 0.545; much better with MT: 'Other': 0.64
* 'TV/Movie Script' (1 instance): 'Opinion/Argumentation': 1.0; the same with MT
* 'Transcript of Video/Audio' (2 instances): 'Information/Explanation': 0.5, 'Other': 0.5; much better with MT: 'Other': 1.0


## Training on the joint schema

## X-GENRE: adding X-CORE datasets