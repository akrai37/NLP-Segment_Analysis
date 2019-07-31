import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#cleaning of dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
'''#removing words 'not' $ 'no' from stopwords
stopwords_remove= stopwords.words('english')
stopword_remove= stopwords_remove.remove('not')
stopword_remove= stopwords_remove.remove('no')'''

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer() 
corpus=[]
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]' , ' ', dataset['Review'][i] )
    review= review.lower()
    review= review.split()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1500)
x= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values 

#Splitting the data into training  set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x, y , test_size=0.20 ,random_state=0)
 
#fitting naiye_bayes classifier into our model
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train , y_train)

#predicting the test set result
y_pred= classifier.predict(x_test)

#confusion_matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix( y_test, y_pred)

#inserting the new values
new_review= "This Place is Awesome"
review= re.sub('[^A-Za-z]',' ', new_review) 
review= review.lower()
review= review.split()
ps= PorterStemmer()
review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review= ' '.join(review)
new_review= [review]
new_review=cv.transform(new_review).toarray() 

#prediction on new review
y_newpred= classifier.predict(new_review)







































    
        