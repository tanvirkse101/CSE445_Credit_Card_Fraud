#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


plt.hist(data['Class'], color='red')
plt.xlabel('Class')
plt.ylabel('Transaction')
plt.title('Class Imbalance', fontsize=15)


# In[6]:


pc_fraud = len(data.loc[data['Class'] == 1].values)/len(data.loc[data['Class'] == 0].values)


# In[7]:


pc_fraud*100


# In[8]:


data.isnull().sum().max()


# In[9]:


isFraudAmt = data.loc[data['Class']==1]['Amount']


# In[10]:


plt.hist(isFraudAmt, color='green')
plt.xlabel('Amount')
plt.ylabel('No. of Transaction')
plt.title('Fraudulent Transaction')


# In[11]:


notFraudAmt = data.loc[data['Class']==0]['Amount']


# In[12]:


plt.hist(notFraudAmt, color='yellow')
plt.hist(isFraudAmt, color='green')
plt.xlabel('Amount')
plt.ylabel('No. of Transaction')
plt.title('Non-Fraudulent Transaction')


# In[13]:


s = sns.boxplot(x="Class", y="Amount", hue="Class",data=data, 
                palette="Set1",showfliers=False).set_title("Transaction Amount")


# In[14]:


plt.hist(data.loc[data['Class']==1]['Time'], edgecolor='white')
plt.xlabel('Time (in seconds)')
plt.ylabel('Frequency')
plt.title('Transaction Time (Fraudulent)')
plt.show()


# In[15]:


plt.hist(data.loc[data['Class']==0]['Time'], edgecolor='white', color='red')
plt.xlabel('Time (in seconds)')
plt.ylabel('Frequency')
plt.title('Transaction Time (Non-Fraudulent)')
plt.show()


# In[3]:


from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()


# In[4]:


data['Norm_Amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))


# In[5]:


data['Norm_Time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))


# In[6]:


data.drop(['Time','Amount'], axis=1, inplace=True)


# In[7]:


data = data.sample(frac=1)


# In[8]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# ### Dimensionality Reduction using PCA

# In[8]:


x = data.drop('Class', axis=1)
y = data['Class']


# In[9]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)


# In[12]:


plt.figure(figsize=(5, 5))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, edgecolor='none',
        cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('PCA on Original Data')
plt.colorbar()


# In[ ]:





# ## Classification with Original Data

# In[22]:


x = data.drop('Class', axis=1).values
y = data['Class'].values


# In[23]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)


# ### Support Vector Machine

# In[24]:


from sklearn.svm import SVC
svc = SVC()


# In[25]:


svc.fit(xtrain, ytrain)


# In[26]:


ypred = svc.predict(xtest)


# In[30]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[31]:


yscore = svc.decision_function(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore)
plot_roc_curve(fpr, tpr)


# ### Random Forest  Classifier

# In[32]:


from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(random_state=0)


# In[33]:


random_clf.fit(xtrain, ytrain)


# In[34]:


ypred = random_clf.predict(xtest)


# In[35]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[44]:


yscore = random_clf.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# In[ ]:





# ### XGBoost

# In[45]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[46]:


xgb.fit(xtrain, ytrain)


# In[47]:


ypred = xgb.predict(xtest)


# In[48]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[51]:


yscore = xgb.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# In[ ]:





# ## Undersampled Data

# In[8]:


fraud_data = data.loc[data['Class']==1]
not_fraud_data = data.loc[data['Class']==0][:492]


# In[9]:


under_data = pd.concat([fraud_data, not_fraud_data])


# In[10]:


under_data = under_data.sample(frac=1, random_state=0)


# In[6]:


plt.hist(under_data['Class'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Under Sampled Data')


# In[7]:


plt.figure(figsize=(24,20))
original_corr = data.corr()
sns.heatmap(original_corr, cmap='Blues')
plt.figure('Confusion Matrix for Orginal Data')


# In[10]:


plt.figure(figsize=(24,20))
under_corr = under_data.corr()
sns.heatmap(under_corr, cmap='Blues')
plt.title('Confusion Matrix for Undersampled Data')
plt.show()


# In[11]:


x = under_data.drop('Class', axis=1)
y = under_data['Class']


# ### Dimensionality Reduction using PCA and t-SNE

# In[12]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)


# In[13]:


plt.figure(figsize=(5, 5))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, edgecolor='none',
        cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('PCA on Undersampled Data')
plt.colorbar()


# In[14]:


from sklearn.manifold import TSNE
x_tsne = TSNE(n_components=2, random_state=0).fit_transform(x)


# In[15]:


plt.figure(figsize=(5, 5))
plt.scatter(x_tsne[:,0], x_tsne[:,1], c=y, edgecolor='none',
        cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('t-SNE on Undersampled Data')
plt.colorbar()


# In[60]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)


# ### Support Vector Machine

# In[61]:


from sklearn.svm import SVC
svc = SVC()


# In[62]:


svc.fit(xtrain, ytrain)


# In[63]:


ypred = svc.predict(xtest)


# In[64]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[65]:


yscore = svc.decision_function(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore)
plot_roc_curve(fpr, tpr)


# ### Random Forest

# In[66]:


from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(random_state=0)


# In[67]:


random_clf.fit(xtrain, ytrain)


# In[68]:


ypred = random_clf.predict(xtest)


# In[69]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[70]:


yscore = random_clf.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# ### XGBoost

# In[71]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[72]:


xgb.fit(xtrain, ytrain)


# In[73]:


ypred = xgb.predict(xtest)


# In[74]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[76]:


yscore = xgb.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# ## Oversampled Data

# In[86]:


not_fraud = data.loc[data['Class']==0]
is_fraud = data.loc[data['Class']==1]


# In[87]:


not_fraud = not_fraud[:(len(not_fraud)//2)]


# In[88]:


is_fraud = is_fraud[:(len(is_fraud)//2)]


# In[89]:


new_data = pd.concat([not_fraud, is_fraud])


# In[90]:


new_data = new_data.sample(frac=1, random_state=0)


# In[91]:


(len(new_data.loc[new_data['Class']==1])/len(new_data.loc[new_data['Class']==0]))*100


# In[92]:


x = new_data.drop('Class', axis=1)
y = new_data['Class']


# In[18]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)


# In[19]:


sm = SMOTE(random_state = 0)
xtrain_over, ytrain_over = sm.fit_sample(xtrain, ytrain.ravel()) 


# ### Dimensionality Reduction using PCA

# In[20]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(xtrain_over)
x_pca = pca.transform(xtrain_over)


# In[22]:


plt.figure(figsize=(5, 5))
plt.scatter(x_pca[:,0], x_pca[:,1], c=ytrain_over, edgecolor='none',
        cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('PCA on Oversampled Data')
plt.colorbar()


# In[30]:


x_df = pd.DataFrame(xtrain_over)
y_df = pd.DataFrame(ytrain_over)


# In[31]:


x_df['Class'] = y_df


# In[32]:


over_data = x_df


# In[33]:


plt.hist(over_data['Class'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Oversampled Data')


# In[43]:


plt.figure(figsize=(24,20))
over_corr = over_data.corr()
sns.heatmap(over_corr, cmap='Blues')
plt.title('Confusion Matrix for Orginal Data')
plt.show()


# ### Support Vector Machine

# In[44]:


from sklearn.svm import SVC
svc = SVC()


# In[45]:


svc.fit(xtrain_over, ytrain_over)


# In[46]:


ypred = svc.predict(xtest)


# In[48]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[49]:


yscore = svc.decision_function(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore)
plot_roc_curve(fpr, tpr)


# In[ ]:





# ### Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(random_state=0)


# In[56]:


random_clf.fit(xtrain_over, ytrain_over)


# In[57]:


ypred = random_clf.predict(xtest)


# In[58]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[59]:


yscore = random_clf.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# ### XGBoost

# In[60]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[61]:


xgb.fit(xtrain_over, ytrain_over)


# In[95]:


xtest_df = pd.DataFrame(xtest)


# In[96]:


xtest_df.columns


# In[97]:


xtest_df.columns = [i for i in range(0, xtest_df.shape[1])]


# In[98]:


xtest = xtest_df.values


# In[100]:


ypred = xgb.predict(xtest)


# In[101]:


print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
print(metrics.classification_report(ytest, ypred))


# In[102]:


yscore = xgb.predict_proba(xtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:,1])
plot_roc_curve(fpr, tpr)


# In[ ]:




