import pandas as pd
import numpy as np
import nltk as nltk
# nltk.download ('all')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn import model_selection,svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class SVM:

	def filteration(self,description):

		# news portal algotrithm
		# CONVERTING INTO LOWERCASE:
		variable_name = description.lower()

		# TOKENIZATION\
		tokens = nltk.word_tokenize(variable_name)
		token_words = [w for w in tokens if w.isalpha()]


			#STOPWORDS:
		stops = set(stopwords.words("english"))
		meaningful_words = [w for w in token_words if not w in stops]


		# STEMMING:
		stemming = nltk.PorterStemmer()
		stemmed_list = [stemming.stem(word) for word in meaningful_words]


		# REJOIN  WORDS:
		joined_words = (",".join(stemmed_list))

		return joined_words




	def categorization(self,descr):

		data = {'datas': [descr]
		}
		text_dataset=pd.DataFrame(data,columns=['datas'])
		text_dataset['datas'] = text_dataset['datas'].apply(self.filteration)
      	#ONLY FOR LRNGTH
		filterationss = self.filteration(descr)
		str1 = str(filterationss)
		desc = str1.split(',',100000)
		length = len(desc)
		print('1')

		#READ CSV 
		corpus =pd.read_csv("final.csv")
		print('2')
		
		#FILTERATION
		corpus['Description'] = corpus['Description'].apply(self.filteration)
		print('3')

			#LABELENCODER
		labelss = LabelEncoder()
		corpus['Label']=labelss.fit_transform(corpus['Label'])

			#TEST_TRAIN_DATA
		X_train,X_test,Y_train,Y_test = train_test_split(corpus['Description'],corpus['Label'],test_size=0.2)
		print('4')

		#TFIDF of TEST_TRAIN data
		Tfidf = TfidfVectorizer(encoding = 'utf-8',max_features = length)                   
		Train_X_Tfidf = Tfidf.fit_transform(X_train).toarray()
		Test_X_Tfidf = Tfidf.transform(X_test).toarray()
		print('5')

		#TFIDF of descr
		Tfidf = TfidfVectorizer(encoding = 'utf-8',
                             max_features= length,
                             )

		descr_Tfidf = Tfidf.fit_transform(text_dataset['datas']).toarray()
		print('6')


		#SVM algorithm part
		SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
		SVM.fit(Train_X_Tfidf,Y_train)
		predictions_SVM = SVM.predict(descr_Tfidf)
		print('7')

		return predictions_SVM

		


svm = SVM()
descr = "boosted by strong car sales. Seasonally adjusted sales rose 1.2% in the month, compared to 0.1% a month earlier, boosted by a surge in shopping just before and after Christmas. Sales climbed 8% for the year, the best performance since an 8.5% rise in 1999, the Commerce Department added. The gains were led by a 4.3% jump in auto sales as dealers used enhanced offers to get cars out of showrooms. Dealers were forced to cut prices in December to maintain sales growth in a tough quarter when the usual end-of-year holiday sales boom was slow to get started. The increase in sales during December pushed total spending for the month to $349.4bn (Â£265.9bn). Sales for the year also broke through the $4 trillion mark for the first time - with annual sales coming in at $4.06 trillion However, if automotives are excluded from December's data, retail sales rose just 0.3% on the month. Home furnishings and furniture stores also performed well, rising 2.2%. But as well as hitting the shops, more US consumers were going online or using mail order for their purchases - with non-store retailers seeing sales rise by 1.9%. However, analysts said that the strong figures were unlikely to put the Federal Reserve Bank off its current policy of measured interest rate rises. Consumers for now remain willing to spend freely, sustaining the US expansion. Given that attitude, the Fed remains likely to continue boosting the Fed funds rate at upcoming meetings, UBS economist Maury Harris told Reuters. Retail sales are seen as a major part of consumer spending - which in turn makes up two-thirds of economic output in the US. Consumer spending has been picking up in recent years after slumping during 2001 and 2002 as the country battled to recover from its first recession of the decade and the World Trade Centre attacks. During that time, sales grew a lacklustre 2.9% in 2001 and 2.5% a year later. Looking ahead, analysts now expect improvement in jobs growth to feed through to the High Street with consumer spending remaining strong. The belief comes despite the latest labor department report showing a surprise rise in unemployment. The number of Americans filing initial jobless claims jumped to 367,000, the highest rate since September. However, long-term claims slipped to their lowest level since 2001."
print(svm.categorization(descr))