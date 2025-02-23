# -*- coding: utf-8 -*-
import nltk
# NLTK includes an example corpus of 3914 tagged sentences
mytaggedsentences=nltk.corpus.treebank.tagged_sents()
myhmmtrainer=nltk.tag.hmm.HiddenMarkovModelTrainer()
# Training using labeled and possibly also unlabeled data
# unfortunately the unsupervised training seems to be buggy!
mytrainedhmm = myhmmtrainer.train( \
    labeled_sequences=mytaggedsentences,unlabeled_sequences=None)
# Alternative: training using "labelled" data only
mytrainedhmm = myhmmtrainer.train_supervised( \
    labelled_sequences=mytaggedsentences)
# Tag a sentence
mytrainedhmm.tag(("The hope is that trained HMM states will " + \
     "correspond to parts of speech.").split())



#%%----------------

pip install hmmlearn
import numpy, hmmlearn, hmmlearn.hmm
# HMMlearn expects the data to be provided 
# as a (nsamples,1) 2D-array, where the 2nd dimension has just
# one element, containing list of indices into a vocabulary, 
# all documents concatenated together, and separately a 
# list of lenghts of the individual documents.
# Create concatenated index list from previously
# crawled and processed documents
concatenated_data=[]
documentlengths=[]
for k in range(len(myindices_in_unifiedvocabulary)):
    concatenated_data.extend(myindices_in_unifiedvocabulary[k])
    documentlengths.append(len(myindices_in_unifiedvocabulary[k]))
concatenated_data=numpy.matrix(concatenated_data).T
# Fit the model
myhmm = hmmlearn.hmm.MultinomialHMM(n_components=10, n_iter=100, verbose=True)
myhmm_fitted=myhmm.fit(concatenated_data,lengths=documentlengths)
# Inspect start, transition, and emission probabilities
myhmm_fitted.startprob_
myhmm_fitted.emissionprob_
myhmm_fitted.transmat_
