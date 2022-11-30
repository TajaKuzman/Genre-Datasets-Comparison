import pandas as pd
import numpy as np
import regex as re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score,precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from statistics import multimode
from apyori import apriori
from sklearn.metrics import accuracy_score
import krippendorff

def label_confidence(corpus, classifiers):
	"""
	Args:
		- classifiers: list of classifiers
		- corpus with predictions
	"""
	# Analyse label confidence per classifier
	min_confidence = []
	max_confidence = []
	median_confidence = []

	for i in classifiers:
		min_confidence.append(round(corpus[f"{i}_confidence"].min(),2))
		max_confidence.append(round(corpus[f"{i}_confidence"].max(),2))
		median_confidence.append(round(corpus[f"{i}_confidence"].median(),2))

	# Confidence dataframe
	confidence = pd.DataFrame({"classifier": classifiers, "min": min_confidence, "median": median_confidence, "max": max_confidence})

	confidence = confidence.sort_values(by="median", ascending=False)
	print(confidence.to_markdown(index=False))

def most_frequent_values(corpus, classifiers):
	"""
	Args:
		- classifiers: list of classifiers
		- corpus with predictions
	"""
	# Compare most frequent values
	most_frequent = []
	frequency = []

	for i in classifiers:
		most_frequent.append(corpus[f"{i}"].mode()[0])
		frequency.append(round(corpus[f"{i}"].value_counts(normalize=True)[0],2))

	most_frequent_comparison = pd.DataFrame({"classifier": classifiers, "most frequent label": list(zip(most_frequent, frequency))})
	print(most_frequent_comparison.to_markdown(index=False))

def label_distribution(corpus, classifiers):
	# Create inital dataframe
	# Create first column
	first_row = list(zip(list(corpus[classifiers[0]].value_counts().to_dict().keys()), [round(x,2) for x in list(corpus[classifiers[0]].value_counts(normalize=True).to_dict().values())]))

	# Append additional elements to the list so that the we will be able to append longer lists in other columns
	while len(first_row) < 12:
		first_row.append("")
	
	new_df = pd.DataFrame({"label_distribution_{}".format(classifiers[0]): first_row})
	
	# Add other columns
	for i in classifiers[1:]:
		next_row = list(zip(list(corpus[i].value_counts().to_dict().keys()), [round(x,2) for x in list(corpus[i].value_counts(normalize=True).to_dict().values())]))
		while len(next_row) < 12:
			next_row.append("")
		new_df["label_distribution_{}".format(i)] =  next_row

	print(new_df.to_markdown(index=False))

# Create a dataframe to analyse the distribution of instance-level labels in domains
def calculate_label_per_domain(corpus, classifiers):
	def genre_analysis(genre_column):
		corpus_analysis_dict = corpus.reset_index().groupby("domain")[genre_column].apply(list).to_dict()
		corpus_analysis_series = pd.Series(list(corpus_analysis_dict.values()), index = list(corpus_analysis_dict.keys()))

		df_items = list(corpus_analysis_dict.values())

		corpus_analysis_df = pd.DataFrame({"domain_id": corpus_analysis_series.index, "genres-in-domain-{}".format(genre_column): [dict(Counter(x)) for x in df_items]})
		return corpus_analysis_df

	# Create a first dataframe
	genre_distribution = genre_analysis(classifiers[0])

	# Create dataframes for all other labels and append them to the first dataframe
	for i in classifiers[1:]:
		new_df = genre_analysis(i)
		genre_distribution = pd.merge(genre_distribution, new_df, how= "left", on= "domain_id", suffixes = ["", ""])
	
	# Add the frequency of the most common label (per domain)

	for i in classifiers:
		label_count_list = list(genre_distribution["genres-in-domain-{}".format(i)])
		most_common_label_frequency = []

		for element in label_count_list:
			# Frequency of the most frequent label
			biggest_number = max(list(element.values()))

			most_common_label_frequency.append(biggest_number)

		genre_distribution["biggest-label-frequency-{}".format(i)] = most_common_label_frequency

	# Add information on the most common label per domain

	for classifier in classifiers:
		# Create a list from the values of the column on biggest label frequency
		counts_list = genre_distribution["biggest-label-frequency-{}".format(classifier)].to_list()

		# Create a list from the values of the column on the genre distribution
		label_distribution_list = genre_distribution["genres-in-domain-{}".format(classifier)].to_list()

		# Create a list of items from the dictionary values
		label_distribution_list_items = [list(x.items()) for x in label_distribution_list]

		# Merge the list of items with the biggest frequency list
		merged_list = list(zip(counts_list, label_distribution_list_items))

		# From this merged list, create a list of all labels that match the biggest frequency value per domain
		frequent_label_list = []

		for merged_element in merged_list:
			# Create a list for each row
			current_most_frequent = []
			# Go through the pairs of labels and their frequency which are in a list in the index 1 of the merged element
			for label_info in merged_element[1]:
				# Compare the frequency of each label with the biggest frequency value (which is in the index 0 of the merged element) - if they are the same, append the label to the list of most frequent labels
				if label_info[1] == merged_element[0]:
					current_most_frequent.append(label_info[0])
			# Append the list of most frequent labels to the global list (for all values)
			frequent_label_list.append(current_most_frequent)

		# Let's create a list of most frequent labels and information whether there was a tie and add them to the dataframe
		tie_list = []
		most_frequent_label_list = []

		for element in frequent_label_list:
			# If there is more than 1 element at the first spot, add to the "tie" list "yes" and randomly choose from the elements which element is added to the most frequent label list
			if len(element) > 1:
				tie_list.append("yes")
				most_frequent_label_list.append(random.choice(element))
			else:
				tie_list.append("no")
				most_frequent_label_list.append(element[0])

		# Add the lists to the dataframe
		genre_distribution["most_frequent_label_{}".format(classifier)] = most_frequent_label_list
		genre_distribution["tie-{}".format(classifier)] = tie_list

	return genre_distribution

# Create graphs based on most common label frequency - for this, we need the domain-level df
def create_graphs(genre_distribution, classifiers, corpus_name):
	"""
	Args:
	- genre_distribution: the dataset where instances are grouped into domains (domain-level dataframe), created with the function "calculate_label_per_domain"
	- classifiers: list of classifiers
	- corpus_name: the name of the corpus
	- save_plot: whether the plots are saved, default is True
	"""
	label_frequency_dict = {}

	for i in classifiers:
		label_frequency_dict[i] = list(genre_distribution["biggest-label-frequency-{}".format(i)])

	sns.set(rc={"figure.figsize": (10, 6)})
	
	for classifier in classifiers:
		ax = sns.kdeplot(label_frequency_dict[classifier], x = range(10), label = "{}".format(classifier), bw_method = 0.25)

	plt.gca().set(title='Frequency of most common label per domain in the sample of {}'.format(corpus_name), ylabel='Frequency of domains', xlabel="Frequency of the most frequent label in domain")
	ax.set_xticks([1,2,3,4,5,6,7,8,9,10])

	plt.legend();

	# Save the plot
	fig1 = plt.gcf()
	plt.show()
	fig1.savefig("Comparison-of-distribution-in-domains-{}-KDE.png".format(corpus_name),dpi=100)

	# Create a histogram

	# Create a dataframe with counts of 1-10 occurences of labels in domains as rows and each classifier as columns

	def create_histogram_df(genre_column):
		histogram_dict = genre_distribution["biggest-label-frequency-{}".format(genre_column)].value_counts().to_dict()
		histogram_series = pd.Series(list(histogram_dict.values()), index = list(histogram_dict.keys()))

		histogram_df = pd.DataFrame({"Occurences of label in domain": histogram_series.index, "number-of-domains-{}".format(genre_column): list(histogram_series)})
		return histogram_df


	# Create a first dataframe
	histogram_df = create_histogram_df(classifiers[0])

	# Create dataframes for all other labels and append them to the first dataframe
	for i in classifiers[1:]:
		new_df = create_histogram_df(i)
		histogram_df = pd.merge(histogram_df, new_df, how= "left", on= "Occurences of label in domain", suffixes = ["", ""])

	# Replace NaNs with 0s
	histogram_df = histogram_df.fillna(value=0)

	# Change floats to integers
	for i in classifiers:
		histogram_df["number-of-domains-{}".format(i)] = histogram_df["number-of-domains-{}".format(i)].astype(int)

	# Sort the histogram based on frequency
	histogram_df = histogram_df.sort_values(by="Occurences of label in domain", ascending=True)

	# Set frequency of labels in domains as the index
	histogram_df.set_index("Occurences of label in domain", inplace=True)

	# Rename the columns
	histogram_df.columns = classifiers

	# Plot the histogram
	plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

	histogram_df.plot(kind='bar', width=0.80, color = ["salmon", "chocolate", "lawngreen", "gold", "olivedrab", "darkviolet", "deepskyblue"])

	plt.legend(loc=2,prop={'size':8})

	plt.gca().set(title='Frequency of most common label per domain in {}'.format(corpus_name), ylabel='No. of domains', xlabel= 'No. of occurences of the most common label in domain')

	# Save the plot
	fig2 = plt.gcf()
	plt.show()
	fig2.savefig("Comparison-of-distribution-in-domains-{}-histogram.png".format(corpus_name),dpi=100)

	# Create separate histograms for each of the classifiers
	# Define the main plot figure that will consists of multiple subplots
	plt.figure(figsize= (8,4))

	# Define the space between the subplots
	plt.subplots_adjust(hspace=.25)

	# First subplot
	position = 0

	colors =["salmon", "chocolate", "lawngreen", "gold", "olivedrab", "darkviolet", "deepskyblue"]

	for i in range(7):
		position += 1
		plt.subplot(4,2,position)
		histogram_df[classifiers[i]].plot(kind='bar', width=0.80,ylim=[0,270], color=colors[i])
		plt.legend(loc=2,prop={'size':7})


	# Save the plot
	fig3 = plt.gcf()
	plt.show()
	fig3.savefig("Comparison-of-distribution-in-domains-{}-subplots.png".format(corpus_name),dpi=100)

# Calculate label distribution per domains
def label_distribution_per_domain(genre_distribution, classifiers):
	# Create inital dataframe
	# Create first column
	first_row = list(zip(list(genre_distribution["most_frequent_label_{}".format(classifiers[0])].value_counts().to_dict().keys()), [round(x,2) for x in list(genre_distribution["most_frequent_label_{}".format(classifiers[0])].value_counts(normalize=True).to_dict().values())]))

	# Append additional elements to the list so that the we will be able to append longer lists in other columns
	while len(first_row) < 12:
		first_row.append("")

	new_df = pd.DataFrame({"label_distribution_{}".format(classifiers[0]): first_row})

	# Add other columns
	for i in classifiers[1:]:
		next_row = list(zip(list(genre_distribution["most_frequent_label_{}".format(i)].value_counts().to_dict().keys()), [round(x,2) for x in list(genre_distribution["most_frequent_label_{}".format(i)].value_counts(normalize=True).to_dict().values())]))
		while len(next_row) < 12:
			next_row.append("")
		new_df["label_distribution_{}".format(i)] =  next_row

	print(new_df.to_markdown(index=False))


# Calculate scores for each classifier, assuming that domains are uni-genre
def scores_based_on_domains(extended_corpus, classifiers):
	# Code for getting micro, macro F1 scores, the classification report and the confusion matrix based on the predicted labels and the most frequent labels per domain (using domain information as a weak signal, under a hypothesis that texts from the same domain are usually in the same genre)

	def scores_based_on_domain_signal(classifier):
		print("Classifier: {}".format(classifier))

		y_pred = extended_corpus["{}".format(classifier)].to_list()
		y_true = extended_corpus["most_frequent_label_{}".format(classifier)].to_list()

		LABELS = list(extended_corpus["{}".format(classifier)].unique())

		# Calculate the scores
		macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
		micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
		accuracy = accuracy_score(y_true, y_pred)
		print("Macro F1: {}, Micro F1: {}, Accuracy: {}".format(round(macro,2), round(micro,2), round(accuracy,2)))

		# Calculate Krippendorff's Alpha

		distr=dict([(b,a) for a,b in enumerate(list(Counter(y_true+y_pred).keys()))])
		y_true_alpha=[distr[e] for e in y_true]
		y_pred_alpha=[distr[e] for e in y_pred]
		k_alpha = krippendorff.alpha(reliability_data=[y_true, y_pred],level_of_measurement='nominal')
		print("Krippendorfs Alpha: {}".format(round(k_alpha, 2)))

		# Print classification report
		print(classification_report(y_true, y_pred, labels = LABELS))
		classification_report_dict = classification_report(y_true, y_pred, labels = LABELS, output_dict=True)

		results = {"classifier":"{}".format(classifier), "Accuracy": round(accuracy, 2), "Krippendorfs Alpha": round(k_alpha, 2), "Macro F1": round(macro,2), "Micro F1": round(micro,2), "Macro precision": round(classification_report_dict['macro avg']['precision'],2), "Macro recall": round(classification_report_dict["macro avg"]["recall"],2)}

		# Plot the confusion matrix:
		cm = confusion_matrix(y_true, y_pred, labels=LABELS)
		plt.figure(figsize=(9, 9))
		plt.imshow(cm, cmap="Oranges")
		for (i, j), z in np.ndenumerate(cm):
			plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
		classNames = LABELS
		plt.ylabel('Predicted label')
		plt.xlabel('Most frequent label in domain')
		tick_marks = np.arange(len(classNames))
		plt.xticks(tick_marks, classNames, rotation=90)
		plt.yticks(tick_marks, classNames)
		plt.title("Comparison of predicted labels and labels that are most frequent per domain: {}".format(classifier))

		plt.tight_layout()
		fig1 = plt.gcf()
		plt.show()
		plt.draw()
		#fig1.savefig("",dpi=100)

		return results

	results_dict= {'Classifier': [], 'Accuracy': [], "Krippendorfs Alpha": [], 'Macro F1': [], 'Micro F1': [], 'Macro precision': [], 'Macro recall': []}

	for classifier in classifiers:
		results_report = scores_based_on_domain_signal(classifier)
		results_dict['Classifier'].append(results_report["classifier"])
		results_dict['Accuracy'].append(results_report["Accuracy"])
		results_dict['Krippendorfs Alpha'].append(results_report["Krippendorfs Alpha"])
		results_dict['Macro F1'].append(results_report['Macro F1'])
		results_dict['Micro F1'].append(results_report['Micro F1'])
		results_dict['Macro precision'].append(results_report['Macro precision'])
		results_dict['Macro recall'].append(results_report['Macro recall'])

	# Create a dataframe with results
	results_df = pd.DataFrame(results_dict)
	results_df = results_df.sort_values("Macro F1", ascending=False)

	print(results_df.to_markdown(index=False))

	return results_dict


# Calculate scores for x-genres based on the majority x-genre label
def scores_based_on_xgenre_majority(extended_corpus, corpus_name):
	x_genre_classifiers = ["GINCO-X-GENRE", "FTD-X-GENRE", "CORE-X-GENRE", "X-GENRE"]

	# Create a list of X-GENRE labels
	majority_label_list = list(zip(list(extended_corpus["GINCO-X-GENRE"]), list(extended_corpus["FTD-X-GENRE"]), list(extended_corpus["CORE-X-GENRE"]), list(extended_corpus["X-GENRE"])))

	# Find the most frequent X-GENRE label out of the 4 X-GENRE labels by using the multimode function
	majority_label_counter_list = [multimode(x) for x in majority_label_list]

	# Create the final list. If there is a tie, add information about that to a specific list and randomly choice the most frequent value
	majority_label = []
	majority_label_tie = []

	for i in majority_label_counter_list:
		if len(i) == 1:
			majority_label.append(i[0])
			majority_label_tie.append("no")
		else:
			majority_label.append(random.choice(i))
			majority_label_tie.append("yes")

	# Add the lists to the dataframe
	extended_corpus["X-GENRE-majority-label"] = majority_label
	extended_corpus["X-GENRE-majority-label-tie"] = majority_label_tie

	# Number of value counts
	print("Number of ties when defining the majority label:")
	print(extended_corpus["X-GENRE-majority-label-tie"].value_counts(normalize=True).to_markdown())

	# Print majority label distribution

	print("Majority label distribution:")

	print(extended_corpus["X-GENRE-majority-label"].value_counts(normalize=True).to_markdown())

	# Overwrite the extended corpus with this additional information
	extended_corpus.to_csv(f"{corpus_name}_with_predictions-domain-info-added.csv", sep="\t")

	# Code for getting micro, macro F1 scores, the classification report and the confusion matrix based on the predicted labels and the majority label
	def scores_based_on_majority_label(classifier):
		print("Classifier: {}".format(classifier))

		y_pred = extended_corpus["{}".format(classifier)].to_list()
		y_true = extended_corpus["X-GENRE-majority-label"].to_list()

		LABELS = list(extended_corpus["{}".format(classifier)].unique())

		# Calculate the scores
		macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
		micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
		accuracy = accuracy_score(y_true, y_pred)
		print("Macro F1: {}, Micro F1: {}, Accuracy: {}".format(round(macro,2), round(micro,2), round(accuracy,2)))

		# Print classification report
		print(classification_report(y_true, y_pred, labels = LABELS))
		classification_report_dict = classification_report(y_true, y_pred, labels = LABELS, output_dict=True)

		results = {"classifier":"{}".format(classifier), "Accuracy": round(accuracy, 2), "Macro F1": round(macro,2), "Micro F1": round(micro,2), "Macro precision": round(classification_report_dict['macro avg']['precision'],2), "Macro recall": round(classification_report_dict["macro avg"]["recall"],2)}

		# Plot the confusion matrix:
		cm = confusion_matrix(y_true, y_pred, labels=LABELS)
		plt.figure(figsize=(7, 7))
		plt.imshow(cm, cmap="Oranges")
		for (i, j), z in np.ndenumerate(cm):
			plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
		classNames = LABELS
		plt.ylabel('Predicted label')
		plt.xlabel('Majority X-GENRE label ("true" label)')
		tick_marks = np.arange(len(classNames))
		plt.xticks(tick_marks, classNames, rotation=90)
		plt.yticks(tick_marks, classNames)
		plt.title("Comparison of predicted labels and the majority X-GENRE label: {}".format(classifier))

		plt.tight_layout()
		fig1 = plt.gcf()
		plt.show()
		plt.draw()

		return results
	
	results_dict_x_genre= {'Classifier': [], 'Accuracy': [], 'Macro F1': [], 'Micro F1': [], 'Macro precision': [], 'Macro recall': []}

	for classifier in x_genre_classifiers:
		results_report = scores_based_on_majority_label(classifier)
		results_dict_x_genre['Classifier'].append(results_report["classifier"])
		results_dict_x_genre['Accuracy'].append(results_report["Accuracy"])
		results_dict_x_genre['Macro F1'].append(results_report['Macro F1'])
		results_dict_x_genre['Micro F1'].append(results_report['Micro F1'])
		results_dict_x_genre['Macro precision'].append(results_report['Macro precision'])
		results_dict_x_genre['Macro recall'].append(results_report['Macro recall'])

	# Create a dataframe with results
	x_genre_df = pd.DataFrame(results_dict_x_genre)
	x_genre_df = x_genre_df.sort_values("Macro F1", ascending=False)

	print(x_genre_df.to_markdown(index=False))
	return results_dict_x_genre

def x_genre_classifier_agreement(extended_corpus):
	x_genre_combinations = [['GINCO-X-GENRE', 'FTD-X-GENRE'], ['GINCO-X-GENRE', 'CORE-X-GENRE'],['GINCO-X-GENRE', 'X-GENRE'], ['FTD-X-GENRE', 'GINCO-X-GENRE'], ['FTD-X-GENRE', 'CORE-X-GENRE'], ['FTD-X-GENRE', 'X-GENRE'], ['CORE-X-GENRE', 'GINCO-X-GENRE'], ['CORE-X-GENRE', 'FTD-X-GENRE'], ['CORE-X-GENRE', 'X-GENRE'], ['X-GENRE', 'GINCO-X-GENRE'], ['X-GENRE', 'FTD-X-GENRE'], ['X-GENRE', 'CORE-X-GENRE']]

	# Code for getting micro, macro F1 scores, the classification report and the confusion matrix based on the predicted labels and the majority label

	def classifier_agreement(classifier_combination):
		print("Comparison of classifiers: {} as y_true, {} as y_pred".format(classifier_combination[0], classifier_combination[1]))
		
		y_true = extended_corpus["{}".format(classifier_combination[0])].to_list()
		y_pred = extended_corpus["{}".format(classifier_combination[1])].to_list()

		# FTD-X-GENRE has less labels (7) than other corpora, so if it occurs in the combination, we will use its labels as the labels list to avoid division by 0
		if classifier_combination[0] == "FTD-X-GENRE" or classifier_combination[1] == 'FTD-X-GENRE':
			LABELS = list(extended_corpus["FTD-X-GENRE"].unique())
		else:
			LABELS = list(extended_corpus["{}".format(classifier_combination[1])].unique())

		# Calculate the scores
		macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
		micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro", zero_division=0)
		print("Macro F1: {}, Micro F1: {}".format(round(macro,2), round(micro,2)))

		# Print classification report
		print(classification_report(y_true, y_pred, labels = LABELS))
		classification_report_dict = classification_report(y_true, y_pred, labels = LABELS, output_dict=True)

		results = {"Classifier as y_true": "{}".format(classifier_combination[0]), "Classifier as y_pred": "{}".format(classifier_combination[1]), "Macro F1": round(macro,2), "Micro F1": round(micro,2), "Macro precision": round(classification_report_dict['macro avg']['precision'],2), "Macro recall": round(classification_report_dict["macro avg"]["recall"],2)}

		# Plot the confusion matrix:
		cm = confusion_matrix(y_true, y_pred, labels=LABELS)
		plt.figure(figsize=(7, 7))
		plt.imshow(cm, cmap="Oranges")
		for (i, j), z in np.ndenumerate(cm):
			plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
		classNames = LABELS
		plt.ylabel('{}'.format(classifier_combination[1]))
		plt.xlabel('{} ("true" label)'.format(classifier_combination[0]))
		tick_marks = np.arange(len(classNames))
		plt.xticks(tick_marks, classNames, rotation=90)
		plt.yticks(tick_marks, classNames)
		plt.title('Comparison of {} as y_true and {} as y_pred'.format(classifier_combination[0], classifier_combination[1]))

		plt.tight_layout()
		fig1 = plt.gcf()
		plt.show()
		plt.draw()
		#fig1.savefig("",dpi=100)

		return results
	
	results_dict_comparison= {"Classifier as y_true": [], "Classifier as y_pred": [], 'Macro F1': [], 'Micro F1': [], 'Macro precision': [], 'Macro recall': []}

	for combination in x_genre_combinations:
		results_report = classifier_agreement(combination)
		results_dict_comparison["Classifier as y_true"].append(results_report["Classifier as y_true"])
		results_dict_comparison["Classifier as y_pred"].append(results_report["Classifier as y_pred"])
		results_dict_comparison['Macro F1'].append(results_report['Macro F1'])
		results_dict_comparison['Micro F1'].append(results_report['Micro F1'])
		results_dict_comparison['Macro precision'].append(results_report['Macro precision'])
		results_dict_comparison['Macro recall'].append(results_report['Macro recall'])

	# Create a dataframe with results
	x_genre_comparison_df = pd.DataFrame(results_dict_comparison)
	x_genre_comparison_df = x_genre_comparison_df.sort_values("Macro F1", ascending=False)

	print(x_genre_comparison_df.to_markdown(index=False))


# Calculate whether the labels match with the apriori rule
def calculate_apriori(extended_corpus, classifiers):

	# Let's use only the relevant columns
	corpus_small = extended_corpus[['FTD', 'GINCO','CORE','GINCO-X-GENRE','FTD-X-GENRE', 'CORE-X-GENRE', 'X-GENRE']]

	# Add information about the schemata to the label names
	for i in classifiers:
		corpus_small[i] = "{}: ".format(i) + corpus_small[i].astype(str)

	# putting the apriori output into a pandas dataframe
	def inspect(output):
		lhs         = [list(result[2][0][0]) for result in output]
		rhs         = [list(result[2][0][1]) for result in output]
		support    = [result[1] for result in output]
		confidence = [result[2][0][2] for result in output]
		lift       = [result[2][0][3] for result in output]
		entire_item = [list(result) for result in output]
		final_lhs = []
		final_rhs = []

		for item in lhs:
			if len(item) > 0:
				final_lhs.append(item[0])
			else:
				final_lhs.append(0)

		for item in rhs:
			if len(item) > 0:
				final_rhs.append(item[0])
			else:
				final_rhs.append(0)
		return list(zip(final_lhs, final_rhs, support, confidence, lift, entire_item))


	def compare_with_apriori(dataframe, column1, column2):
		apriori_list =  list(list(x) for x in zip(list(dataframe[column1]), list(dataframe[column2])))

		results = list(apriori(apriori_list,
				min_support=0.01,
				min_confidence=0.50,
				min_lift=1.0,
				max_length=None))

		output_df = pd.DataFrame(inspect(list(results)), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift','Full_result'])

		output_df = output_df.sort_values(by="Lift", ascending=False)

		# Filter out values with 0 at lhs or rhs
		output_df = output_df[output_df["Left_Hand_Side"] != 0]
		output_df = output_df[output_df["Right_Hand_Side"] != 0]

		return output_df
	
	# Let's compare with apriori
	for i in classifiers:
		for i2 in classifiers:
			if i != i2:
				print("Comparison: {} with {}".format(i,i2))
				df = compare_with_apriori(corpus_small, i, i2).drop(columns="Full_result")
				print(df.to_markdown(index = False))

				# Calculate which labels are missing
				not_matched = []

				for label in list(corpus_small[i].unique()):
					if label not in list(df["Left_Hand_Side"].unique()) and label not in list(df["Right_Hand_Side"].unique()):
						not_matched.append(label)
				for label in list(corpus_small[i2].unique()):
					if label not in list(df["Left_Hand_Side"].unique()) and label not in list(df["Right_Hand_Side"].unique()):
						not_matched.append(label)
				print(f"Labels not matched: {not_matched}")