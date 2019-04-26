import csv
import os
import pandas as pd
from dask.threaded import get
from dask import compute, delayed
import matplotlib.pyplot as plt
import numpy as np

keywords = []
clusters = []
clusterNames = []


def visualise_summary(summary_frame):
    if not os.path.isdir("Charts"):
        os.mkdir("Charts")
    # Plotting fin-tech - non fintech percentage chart
    x = summary_frame[['Bank Name', 'FinTech Score', 'Non-Fintech Score']]
    y = x.set_index('Bank Name')
    z = y.groupby('Bank Name').mean()

    z.plot(stacked=True, figsize=(15, 8), kind='barh')

    df_total = summary_frame['FinTech Score'] + summary_frame['Non-Fintech Score']
    df_rel = summary_frame[summary_frame.columns[1:3]].div(df_total, 0) * 100

    for n in df_rel:
        for i, (cs, ab, pc, tot) in enumerate(
                zip(summary_frame.iloc[:, 1:].cumsum(1)[n], summary_frame[n], df_rel[n], df_total)):
            plt.text(tot, i, str(tot), va='center')
            plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', rotation=45)

    plt.savefig('Charts/Fintech-Nonfintech.png')

    # Plotting cluster concentration in each bank
    # summary_frame.T.plot.pie(subplots=True, figsize=(10, 3))
    bankNames = list(summary_frame['Bank Name'])

    x = summary_frame[['Admin_Clerical_HR - Sheet1', 'Audit and Finance', 'Business Intelligence and Analysis',
                       'Data Analytics and Machine Learning', 'Marketing , Sales and management',
                       'Network and Cyber Security', 'Softwrae Development',
                       'Investment and risk management_ - Sheet1']]
    count = 0
    for ind in x.index:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        pie = x.iloc[ind].plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        ax.set_xlabel('')
        figure = pie.get_figure()
        figure.savefig('Charts/' + bankNames[count] + '.png')
        count += 1


def analyse_data(threads):
    print("Preparing data to visualize")
    headers = ['Bank Name', 'FinTech Score', 'Non-Fintech Score']
    headers.extend(clusterNames)
    bank_summary = {header: [] for header in headers}
    for filename in os.listdir("ClusterScore"):
        if filename.endswith(".csv"):
            bank_summary['Bank Name'].append(filename.split('.')[0])
            bank = pd.read_csv("ClusterScore/" + filename)
            col_list = list(bank)
            col_list.remove('Bank Name')
            col_list.remove('Job Title')
            col_list.remove('URL (URL of the job posting)')
            col_list.remove('Unnamed: 0')

            for key, value in bank_summary.items():
                if (key != 'Bank Name'):
                    bank_summary[key].append(bank[col_list].sum(axis=0)[key])

    summary_frame = pd.DataFrame(bank_summary, columns=bank_summary.keys())
    visualise_summary(summary_frame)


def get_cluster(keyword):
    for cluster in clusters:
        for key, value in cluster.items():
            if (keyword in value):
                return key
    return "None"


def write_cluster_to_csv(fileName, urlDict):
    if not os.path.isdir("ClusterScore"):
        os.mkdir("ClusterScore")
    if (type(urlDict) is dict):
        data = pd.DataFrame([urlDict], columns=urlDict.keys())
        # if file does not exist write header
        if not os.path.isfile('ClusterScore/' + fileName):
            data.to_csv('ClusterScore/' + fileName, header='column_names')
        else:  # else it exists so append without writing the header
            data.to_csv('ClusterScore/' + fileName, mode='a', header=False)
    else:
        print("exception?")


def calculate_clusters_scores(rowDict):
    finTechClusters = ['Data Analytics and Machine Learning', 'Network and Cyber Security', 'Softwrae Development']
    for key, value in rowDict.items():
        if (key in clusterNames and value > 0):
            if (key in finTechClusters):
                score = rowDict['FinTech Score']
                newScore = score + 1
                rowDict['FinTech Score'] = newScore
            else:
                score = rowDict['Non-Fintech Score']
                newScore = score + 1
                rowDict['Non-Fintech Score'] = newScore
    return rowDict


def classify_bank(fileName):
    dict_list = []
    print("Classifying postings of - " + fileName.split('.')[0])
    with open("Output/" + fileName, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        d = {name: [] for name in reader.fieldnames}
        for row in reader:
            rowDict = {'Bank Name': '', 'Job Title': '', 'URL (URL of the job posting)': '', 'FinTech Score': 0,
                       'Non-Fintech Score': 0}
            for cluster in clusterNames:
                rowDict[cluster] = 0
            for name in reader.fieldnames:
                if (name in ['Bank Name', 'Job Title', 'URL (URL of the job posting)']):
                    rowDict[name] = row[name]
                else:
                    if (int(row[name]) > 0):
                        cluster = get_cluster(name)
                        if (cluster != "None"):
                            rowDict[cluster] = 1
            write_cluster_to_csv(fileName, calculate_clusters_scores(rowDict))


def write_wordCount_to_csv(fileName, urlDict):
    if not os.path.isdir("Output"):
        os.mkdir("Output")
    if (type(urlDict) is dict):
        data = pd.DataFrame([urlDict], columns=urlDict.keys())
        # if file does not exist write header
        if not os.path.isfile('Output/' + fileName):
            data.to_csv('Output/' + fileName, header='column_names')
        else:  # else it exists so append without writing the header
            data.to_csv('Output/' + fileName, mode='a', header=False)
    else:
        print("exception?")


def scrape_file(fileName):
    print("Analyzing postings of - " + fileName.split('.')[0])
    with open("Data/" + fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            metaDict = {"Bank Name": "", "Job Title": "", "URL (URL of the job posting)": ""}
            wordCountDict = dict.fromkeys(keywords, 0)
            metaDict["Bank Name"] = row[0]
            metaDict["Job Title"] = row[2]
            metaDict["URL (URL of the job posting)"] = row[3]
            desc_words = row[1].split()
            desc_bigrams = [desc_words[i] + " " + desc_words[i + 1] for i in range(0, len(desc_words) - 1)]
            for word in keywords:
                if (word in desc_bigrams):
                    wordCountDict[word] = desc_bigrams.count(word)
                else:
                    wordCountDict[word] = desc_words.count(word)
            write_wordCount_to_csv(fileName, {**metaDict, **wordCountDict})
    classify_bank(fileName)


def get_all_clusters():
    print("Getting clusters")
    for filename in os.listdir("Clusters"):
        if filename.endswith(".csv"):
            with open("Clusters/" + filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                keywords = list(reader)
            clusters.append({filename.split('.')[0]: [i[0] for i in keywords]})
            clusterNames.append(filename.split('.')[0])


def get_fileNames():
    print("Getting bank names")
    fileNames = []
    for filename in os.listdir("Data"):
        if filename.endswith(".csv"):
            fileNames.append(filename)
    return fileNames


def get_keywords():
    print("Getting keywords")
    with open("tfidf_textrank_wordcount.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        kw = list(reader)
    keywords.extend([i[0] for i in kw])


dsk1 = {'get_keywords': get_keywords(),
        'get_all_clusters': get_all_clusters()}

dsk2 = {'get_meta': compute(*[delayed(process for key, process in dsk1.items())], scheduler='single-threaded'),
        'create_threads': (
        compute(*[delayed(scrape_file)(x) for x in get_fileNames()], scheduler='threads'), 'get_meta'),
        'analyse_data': (analyse_data, 'create_threads')}

if _name_ == '_main_':
    get(dsk2, 'analyse_data')