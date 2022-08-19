
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
* 'How-To/Instructional': 'A7 (instruction)': 0.713 (percentage of instances of How-To/Instructional, identified as A7)

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

#### FTD to CORE-sub categories

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

#### CORE-sub to FTD categories

Most of the FTD categories are well connected to specific CORE-sub categories.

1. Well-connected labels:
* 'A11 (personal)': 'Personal Blog': 0.48
* 'A14 (academic)': 'Research Article': 0.57
* 'A16 (information)': 'Description of a Thing': 0.47, 'Encyclopedia Article': 0.17
* 'A17 (review)': 'Reviews': 0.53
* 'A4 (fiction)': 'Short Story': 0.78
* 'A8 (news)': 'News Report/Blog': 0.68
* 'A9 (legal)': 'Legal terms': 0.84

2. Not connected labels:
* 'A1 (argumentative)': 'Description of a Thing': 0.25, scattered across all labels
* 'A12 (promotion)': 'Description of a Thing': 0.55, 'Description with Intent to Sell': 0.25
* 'A7 (instruction)': 'How-to': 0.42, 'Description of a Thing': 0.19

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


#### GINCO/MT-GINCO to CORE-sub

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
* 'Reviews': 'Opinion/Argumentation': 0.5, 'Promotion': 0.279; much better with MT: 'Opinion/Argumentation': 0.647
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

#### CORE-sub to GINCO/MT-GINCO

On 249 instances (25%) are the CORE-sub labels predicted to Slovene text different than those predicted on the MT text.

Half of the 24 GINCO categories are well connected to the CORE subcategories (well predicted by the CORE-sub classifier). In most cases, prediction on MT-GINCO improves the results (FAQ - 67 points better, Instruction - 8 points better, Song Lyrics - 25 points better, News/Reporting - 6 points better, Recipe - 17 points better, Research Article - 33 points better), for some categories, the predictions were worse (Forum - 8 points worse, Legal Terms - 6 points worse, Promotion of a Product - 1 point worse, Review - 12 points worse). For some GINCO labels 100% of the instances were correctly predicted by the CORE-sub labels: 'FAQ': 'FAQ about Information'(on MT), 'Interview': 'Interview' (on both Slovene and MT), 'Recipe': 'Recipe' (on MT), 'Research Article': 'Research Article' (on MT).

Despite the fact that Description of a Thing represents 9% of the instances in CORE-sub dataset, most categories that were hard to identify were predicted this label. This suggests that the label is very fuzzy and can thus incorporate so many different genres. Very rare labels, (Course Materials, Formal Speech, Magazine Article, Travel Blog, Persuasive Article/Essay, Transcript of Video/Audio, Technical Support, Other Information, Reader/Viewer Responses, Editorial, FAQ about How-To, Technical Report, Letter to Editor), were not predicted to any instance in the GINCO dataset.

If there is no information regarding the MT-GINCO results, they are the same as the GINCO.

1. Well connected labels:
* 'FAQ': 'Description of a Thing': 0.33, 'FAQ about Information': 0.33, 'Question/Answer Forum': 0.33; much better on MT-GINCO: 'FAQ about Information': 1.0
* 'Forum': 'Discussion Forum': 0.807, a bit worse on MT-GINCO: 'Discussion Forum': 0.73
* 'Information/Explanation': 'Description of a Thing': 0.6, 'Encyclopedia Article': 0.11, similar on MT
* 'Instruction': 'How-to': 0.42, 'Description of a Thing': 0.21; better on MT: 'How-to': 0.5, 'Description of a Thing': 0.21
* 'Interview': 'Interview': 1.0
* 'Legal/Regulation': 'Legal terms': 0.88, a bit worse on MT: 'Legal terms': 0.82
* 'Lyrical': 'Short Story': 0.5, 'Religious Blogs/Sermons': 0.25, 'Song Lyrics': 0.25; better on MT: 'Song Lyrics': 0.5, 'Religious Blogs/Sermons': 0.25, 'Short Story': 0.25
* 'News/Reporting': 'News Report/Blog': 0.49, 'Description of a Thing': 0.23, 'Sports Report': 0.11; better on MT: 'News Report/Blog': 0.55, 'Description of a Thing': 0.25, 'Sports Report': 0.10
* 'Promotion of a Product': 'Description with Intent to Sell': 0.50, 'Description of a Thing': 0.32; a bit worse on MT: 'Description with Intent to Sell': 0.49, 'Description of a Thing': 0.35
* 'Prose': 'Short Story': 0.83
* 'Recipe': 'Recipe': 0.83; even better on MT: 'Recipe': 1.0
* 'Research Article': 'Research Article': 0.67; even better on MT: 'Research Article': 1.0

2. Not well connected: 
* Announcement: Description of a Thing: 0.65
* 'Call': 'Description of a Thing': 0.64, 'Legal terms': 0.36; MT: 'Description of a Thing': 0.73, 'Legal terms': 0.18
* Correspondence: 'Question/Answer Forum': 0.38, scattered over multiple categories; MT: 'Question/Answer Forum': 0.44, scattered across multiple categories
* 'Invitation': 'Description of a Thing': 0.97, similar on MT
* 'List of Summaries/Excerpts': scattered across all categories
* 'Opinion/Argumentation': scattered across all categories: 'Description of a Thing': 0.19, 'Interview': 0.15, 'Personal Blog': 0.11 and others; similar on MT
* 'Opinionated News': 'News Report/Blog': 0.33, 'Sports Report': 0.29, 'Description of a Thing': 0.22; a bit better on MT: 'News Report/Blog': 0.40, 'Sports Report': 0.27
* 'Other': scattered across categories, 'Description of a Thing': 0.44; similar on MT
* 'Promotion': 'Description of a Thing': 0.67, 'Description with Intent to Sell': 0.17; similar on MT
* 'Promotion of Services': 'Description of a Thing': 0.78, 'Description with Intent to Sell': 0.125; similar on MT
* 'Review': 'Reviews': 0.41, 'Personal Blog': 0.18, 'Description of a Thing': 0.18; even worse on MT: 'Reviews': 0.29, scattered across multiple categories
* 'Script/Drama': 'Encyclopedia Article': 1.0, the same on MT
