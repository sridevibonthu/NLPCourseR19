#E-mail spam prediction using NB

import pandas as pd 

#read the data
sms_data = pd.read_csv('SMSSpamCollection', header=None, sep='\t', names=['Label', 'Sms'])

print(f'The shape of the data {sms_data.shape}')

print(sms_data.groupby('Label').count())

#Data Preparation
sms_data['Sms'] = sms_data['Sms'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
sms_data['Sms'] = sms_data['Sms'].str.lower()
sms_data['Sms'] = sms_data['Sms'].str.split()

#train-test split
#train test split
train_data = sms_data.sample(frac=0.8,random_state=1).reset_index(drop=True)
test_data = sms_data.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

print('Records of training data : ', len(train_data), '\nRecords of test data : ', len(test_data))

#prepare the vocabulary and count the number of separate words in each message
vocabulary = list(set(train_data['Sms'].sum()))
word_counts_per_sms = pd.DataFrame([
    [row[1].count(word) for word in vocabulary]
    for _, row in train_data.iterrows()], columns=vocabulary)
train_data = pd.concat([train_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]

print(train_data.head(2))

# Implementation

# probability of message to be spam
Pspam = train_data['Label'].value_counts()['spam'] / train_data.shape[0]
#probability of non-spam messages
Pham = train_data['Label'].value_counts()['ham'] / train_data.shape[0]
#the number of words in spam messages
Nspam = train_data.loc[train_data['Label'] == 'spam', 'Sms'].apply(len).sum()
#the number of words in ham messages
Nham = train_data.loc[train_data['Label'] == 'ham', 'Sms'].apply(len).sum()
Nvoc = len(train_data.columns) - 3
alpha = 1

def p_w_spam(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['Label'] == 'spam', word].sum() + alpha) / (Nspam + alpha*Nvoc)
    else:
        return 1
      
def p_w_ham(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['Label'] == 'ham', word].sum() + alpha) / (Nham + alpha*Nvoc)
    else:
        return 1
        

def classify(message):
    p_spam_given_message = Pspam
    p_ham_given_message = Pham
    for word in message:
        p_spam_given_message *= p_w_spam(word)
        p_ham_given_message *= p_w_ham(word)
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'needs human classification'


print('The message secret is classified as ', classify('secret'))

print("Evaluation on test data : ")

#predictions on test data
test_data['predicted'] = test_data['Sms'].apply(classify)
correct = (test_data['predicted'] == test_data['Label']).sum() / test_data.shape[0] * 100
print("Accuracy : ", correct)