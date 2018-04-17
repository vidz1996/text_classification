from django.shortcuts import render
from text_classifier.forms import Text_Form
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib



def model_train():
    COLUMN_NAME = ['Category', 'Content']
    pd.options.display.max_rows = 10

    train_data = pd.read_csv("./data/train.csv", names=COLUMN_NAME)
    X_train, y_train = train_data, train_data.pop('Category')

    test_data = pd.read_csv("./data/test.csv", names=COLUMN_NAME)
    X_test, y_test = test_data, test_data.pop('Category')
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit(X_train['Content']) 
    X_train_counts = count_vect.transform(X_train['Content']) 
    transformer = TfidfTransformer()
    X_train_tfidf = transformer.fit_transform(X_train_counts) 
    X_train_tfidf.shape #x

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    joblib.dump(count_vect.vocabulary_,'vocabulary.pkl')
    joblib.dump(clf, 'train_model.pkl') 

    

def index(request):
    clf = joblib.load('train_model.pkl')
    vocabulary = joblib.load('vocabulary.pkl') 
    load_vectorizer = CountVectorizer(vocabulary=vocabulary)
    CATEGORY = ['Animals','Beauty & Style','Business & Finance','Technology','Education','Entertainment','Environment','Food & Drink','Health & Medicines','Sports']
    #CATEGORY = ['World', 'Sports', 'Business', 'Sci/Tech']
    #CATEGORY = ['Business and Companies','Educational Institution','Artist','Sports','Politics','Mean Of Transportation','Buildings and Monuments','Natural Places','Village','Animal','Plant','Music','Film','Literature and Poetry','Technology']
    context_dict={}
    context_dict['cat_tree']=CATEGORY
    form = Text_Form()
    l1=[]
    if request.method=='POST':
        form = Text_Form(request.POST)
        if form.is_valid():
            print("========form===========")
            print(form)
            print("=====================")
            line = request.POST.get("text")
            print("========Line===========")
            print(type(line))
            context_dict['line'] = line
            print("=====================")
            l1.append(line)   
            X_new_counts = load_vectorizer.transform(l1)
            print("\n")
            predicted = clf.predict(X_new_counts)
            for i in predicted:
                a=int(i)
                category=CATEGORY[a-1]
                print(category)
                context_dict['category']=category
        else:
            print(form.errors)
    return render(request,"index.html",context_dict)