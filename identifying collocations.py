# -*- coding: utf-8 -*-
import pandas as pd
import requests 
from bs4 import BeautifulSoup
import bs4
import requests
import nltk
import numpy
import matplotlib.pyplot

#%% Vocabulary pruning
nltkstopwords=nltk.corpus.stopwords.words('english')
pruningdecisions=numpy.zeros((len(unifiedvocabulary),1))
for k in range(len(unifiedvocabulary)):
    # Rule 1: check the nltk stop word list
    if (unifiedvocabulary[k] in nltkstopwords):
        pruningdecisions[k]=1
    # Rule 2: if the word is in the top 1% of frequent words
    if (k in highest_totaloccurrences_indices[\
        0:int(numpy.floor(len(unifiedvocabulary)*0.01))]):
        pruningdecisions[k]=1
    # Rule 3: if the word is in the bottom 65% of frequent words
    if (k in highest_totaloccurrences_indices[(int(numpy.floor(\
        len(unifiedvocabulary)*0.35))):len(unifiedvocabulary)]):
        pruningdecisions[k]=1
    # Rule 4: if the word is too short
    if len(unifiedvocabulary[k])<2:
        pruningdecisions[k]=1
    # Rule 5: if the word is too long
    if len(unifiedvocabulary[k])>20:
        pruningdecisions[k]=1
    # Rule 6: if the word has unwanted characters
    # (here for simplicity only a-z allowed)
    if unifiedvocabulary[k].isalpha()==False:
        pruningdecisions[k]=1



# ----------------------------------------------------------



#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(unifiedvocabulary)):
    if pruningdecisions[k]==0:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)

#%% Create pruned texts

mycrawled_prunedtexts=[]
myindices_in_prunedvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
    print(k)
    temp_newindices=[]
    temp_newdoc=[]
    for l in range(len(mycrawled_lemmatizedtexts[k])):
        temp_oldindex=myindices_in_unifiedvocabulary[k][l]            
        temp_newindex=oldtopruned[temp_oldindex]
        if temp_newindex!=-1:
            temp_newindices.append(temp_newindex)
            temp_newdoc.append(unifiedvocabulary[temp_oldindex])
    mycrawled_prunedtexts.append(temp_newdoc)
    myindices_in_prunedvocabulary.append(temp_newindices)



# ----------------------------------------------------------



#%% Inspect remaining frequent words
# Sort remaining words by largest total (or mean) occurrence count
remainingindices=numpy.squeeze(numpy.where(pruningdecisions==0)[0])
remainingvocabulary=unifiedvocabulary[remainingindices]
remainingvocabulary_totaloccurrencecounts= \
    unifiedvocabulary_totaloccurrencecounts[remainingindices]
remaining_highest_totaloccurrences_indices= \
    numpy.argsort(-1*remainingvocabulary_totaloccurrencecounts,axis=0)
print(numpy.squeeze(remainingvocabulary[remaining_highest_totaloccurrences_indices[1:500]]))
print(numpy.squeeze(remainingvocabulary_totaloccurrencecounts[ \
    remaining_highest_totaloccurrences_indices[1:500]]))



# ----------------------------------------------------------



mycrawled_lemmatizedtexts[1].dispersion_plot(['research','university','study'])



# ----------------------------------------------------------



mycrawled_lemmatizedtexts[0].concordance('research')



# ----------------------------------------------------------



#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(unifiedvocabulary)):
    if pruningdecisions[k]==0:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)
#%% Create pruned texts
mycrawled_prunedtexts=[]
myindices_in_prunedvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
    print(k)
    temp_newindices=[]
    temp_newdoc=[]
    for l in range(len(mycrawled_lemmatizedtexts[k])):
        temp_oldindex=myindices_in_unifiedvocabulary[k][l]            
        temp_newindex=oldtopruned[temp_oldindex]
        if temp_newindex!=-1:
            temp_newindices.append(temp_newindex)
            temp_newdoc.append(unifiedvocabulary[temp_oldindex])
    mycrawled_prunedtexts.append(temp_newdoc)
    myindices_in_prunedvocabulary.append(temp_newindices)



# ----------------------------------------------------------



#%% Compute statistics of word distances
# Compute counts and subs of distances and squared distances
import scipy
distanceoccurrences=scipy.sparse.lil_matrix(\
    (len(remainingvocabulary),len(remainingvocabulary)))
sumdistances=scipy.sparse.lil_matrix(\
    (len(remainingvocabulary),len(remainingvocabulary)))
sumabsdistances=scipy.sparse.lil_matrix(\
    (len(remainingvocabulary),len(remainingvocabulary)))
sumdistancesquares=scipy.sparse.lil_matrix(\
    (len(remainingvocabulary),len(remainingvocabulary)))



# ----------------------------------------------------------



for l in range(len(mycrawled_lemmatizedtexts)):
    latestoccurrencepositions=scipy.sparse.lil_matrix(\
        (len(remainingvocabulary),len(remainingvocabulary)))
    # Loop through all word positions m of document l
    for m in range(len(mycrawled_prunedtexts[l])):
        # Get the vocabulary index of the current word in position m
        currentword=myindices_in_prunedvocabulary[l][m]
        # Loop through previous words, counting back up to 10 words from current word
        windowsize=min(m,10)
        for n in range(windowsize):
            # Get the vocabulary index of the previous word in position m-n-1
            previousword=myindices_in_prunedvocabulary[l][m-n-1]
            # Is this the fist time we have encountered this word while
            # counting back from the word at m? Then it is the closest pair.
            if latestoccurrencepositions[currentword,previousword]<m:            
                # Store the occurrence of this word pair with the word at m as the 1st word
                distanceoccurrences[currentword,previousword]=\
                    distanceoccurrences[currentword,previousword]+1
                sumdistances[currentword,previousword]=sumdistances[\
                    currentword,previousword]+((m-n-1)-m)
                sumabsdistances[currentword,previousword]=\
                    sumabsdistances[currentword,previousword]+abs((m-n-1)-m)
                sumdistancesquares[currentword,previousword]=\
                    sumdistancesquares[currentword,previousword]+((m-n-1)-m)**2
                # Store the occurrence of this word pair with the word at n as the 1st word
                distanceoccurrences[previousword,currentword]=\
                    distanceoccurrences[previousword,currentword]+1
                sumdistances[previousword,currentword]=sumdistances[\
                    previousword,currentword]+(m-(m-n-1))
                sumabsdistances[previousword,currentword]=\
                    sumabsdistances[previousword,currentword]+abs(m-(m-n-1))
                sumdistancesquares[previousword,currentword]=\
                    sumdistancesquares[previousword,currentword]+(m-(m-n-1))**2
                # Mark that we found this pair while counting down from m,
                # so we do not count more distant occurrences of the pair
                latestoccurrencepositions[currentword,previousword]=m
                latestoccurrencepositions[previousword,currentword]=m



# ----------------------------------------------------------



# Compute distribution statistics based on the counts
n_vocab=len(remainingvocabulary)
distancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
distancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
for m in range(n_vocab):
    print(m)
    for n in range(n_vocab):
        n_occurrences=distanceoccurrences[m,n]
        if n_occurrences>1:
            # Estimate mean of m-n distance
            distancemeans[m,n]=sumdistances[m,n]/n_occurrences
            absdistancemeans[m,n]=sumabsdistances[m,n]/n_occurrences
            # Estimate variance of m-n distance
            distancevariances[m,n]=\
                sumdistancesquares[m,n]/(n_occurrences-1) \
                - (n_occurrences/(n_occurrences-1))\
                *(distancemeans[m,n]**2)
            absdistancevariances[m,n]=\
                sumdistancesquares[m,n]/(n_occurrences-1) \
                - (n_occurrences/(n_occurrences-1))\
                *(absdistancemeans[m,n]**2)



# ----------------------------------------------------------



# Compute distribution statistics based on the counts
n_vocab=len(remainingvocabulary)
distancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
distancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
for m in range(n_vocab):
    print(m)
    # Find the column indices that have at least two occurrences
    tempindices=numpy.nonzero(distanceoccurrences[m,:]>1)[1]
    # The occurrence vector needs to be a non-sparse data type
    tempoccurrences=distanceoccurrences[m,tempindices].todense()
    # Estimate mean of m-n distance
    distancemeans[m,tempindices]=numpy.squeeze(\
        numpy.array(sumdistances[m,tempindices]/tempoccurrences))
    absdistancemeans[m,tempindices]=numpy.squeeze(\
        numpy.array(sumabsdistances[m,tempindices]/tempoccurrences))
    # Estimate variance of m-n distance
    meanterm=distancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    distancevariances[m,tempindices]=numpy.squeeze(\
        numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) \
        - meanterm))
    meanterm=absdistancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    absdistancevariances[m,tempindices]=numpy.squeeze(\
        numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) \
        - meanterm))



# ----------------------------------------------------------



# Compute overall distance distribution
overalldistancecount=numpy.sum(distanceoccurrences)
overalldistancesum=numpy.sum(sumdistances)
overallabsdistancesum=numpy.sum(sumabsdistances)
overalldistancesquaresum=numpy.sum(sumdistancesquares)
overalldistancemean=overalldistancesum/overalldistancecount
overallabsdistancemean=overallabsdistancesum/overalldistancecount
overalldistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
    -overalldistancecount/(overalldistancecount-1)*(overalldistancemean**2)
overallabsdistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
    -overalldistancecount/(overalldistancecount-1)*(overallabsdistancemean**2)



# ----------------------------------------------------------



def findwordindex(wordstring):
    for k in range(len(remainingvocabulary)):
        if remainingvocabulary[k]==wordstring:
            return(k)
    return(-1)

# Find the chosen word and words that occurred with it at least 2 times
mywordindex=findwordindex('developer')
tempindices=numpy.nonzero(distanceoccurrences[mywordindex,:]>1)[1]

# Sort the pairs by lowest mean absolute distance
lowest_meandistances_indices=numpy.argsort(numpy.squeeze(numpy.array(\
    absdistancemeans[mywordindex,tempindices].todense())),axis=0)

# Print the top-50 lowest-distance pairs
for k in range(50):
    otherwordindex=tempindices[lowest_meandistances_indices[k]]
    # Print word pairs, absolute distances and distances (mean+std)
    print((remainingvocabulary[mywordindex],\
        remainingvocabulary[otherwordindex],\
        absdistancemeans[mywordindex,otherwordindex],\
        numpy.sqrt(absdistancevariances[\
        mywordindex,otherwordindex]),\
        distancemeans[mywordindex,otherwordindex],\
        numpy.sqrt(distancevariances[mywordindex,otherwordindex])))


# ----------------------------------------------------------



import scipy.stats
pvalue = 1 - stats.t.cdf(d,df=df)



# ----------------------------------------------------------



#%% Compute t-test pvalues comparing abs distance distributions
absdistancepvalues=scipy.sparse.lil_matrix((n_vocab,n_vocab))

for m in range(n_vocab):
    # Find pairs of word m
    tempindices=numpy.nonzero(distanceoccurrences[m,:]>1)[1]
    # For computation we need to transform these to non-sparse vectors
    meanterm=absdistancemeans[m,tempindices].todense()
    varianceterm=absdistancevariances[m,tempindices].todense()
    occurrenceterm=distanceoccurrences[m,tempindices].todense()

    # Compute the t-test statistic for each pair
    tempstatistic=(meanterm-overallabsdistancemean)/ \
        numpy.sqrt(varianceterm/occurrenceterm+ \
        overallabsdistancevariance/overalldistancecount)
    # Compute the t-test degrees of freedom for each pair
    tempdf=(numpy.power(varianceterm/occurrenceterm+\
        overallabsdistancevariance/overalldistancecount,2))/ \
        ( (numpy.power(varianceterm/occurrenceterm,2))/(occurrenceterm-1)+ \
        ((overallabsdistancevariance/overalldistancecount)**2)/ \
        (overalldistancecount-1) )
    # Compute the t-test p-value for each pair
    temppvalue=1-scipy.stats.t.cdf(tempstatistic,tempdf)
    # Store the t-test p-value for each pair
    absdistancepvalues[m,tempindices]=numpy.squeeze(numpy.array(temppvalue))



# ----------------------------------------------------------



#%% Sort word pairs of a particular word by minimum mean absolute distance
def findwordindex(wordstring):
    for k in range(len(remainingvocabulary)):
        if remainingvocabulary[k]==wordstring:
            return(k)
    return(-1)
    
def printtopcollocations(wordstring):
    # Find the chosen word and words that occurred with it at least 2 times     
    mywordindex=findwordindex(wordstring)
    if mywordindex==-1:
        print('Word not found: '+wordstring)
        return
    # Require at least 10 pair occurrences
    minpairoccurrences=10
    tempindices=numpy.nonzero(distanceoccurrences[mywordindex,:]>minpairoccurrences)[1]

    # Sort the pairs by lowest pvalue
    lowest_meandistances_indices=numpy.argsort(numpy.squeeze(\
        numpy.array(absdistancepvalues[mywordindex,tempindices].todense())),axis=0)
    # Print the top-50 lowest-distance pairs
    print('\nLowest p-values\n')
    for k in range(min(50,len(lowest_meandistances_indices))):
        otherwordindex=tempindices[lowest_meandistances_indices[k]]
        # Print the words, their absolute distances (mean+std) and distances (mean+std)
        print('{!s}--{!s}: {:d} occurrences, absdist: {:.1f} +- {:.1f}, offset: {:.1f} +- {:.1f}, pvalue: {:f}'.format(\
               remainingvocabulary[mywordindex],\
               remainingvocabulary[otherwordindex],\
               int(distanceoccurrences[mywordindex,otherwordindex]),\
               absdistancemeans[mywordindex,otherwordindex],\
               numpy.sqrt(absdistancevariances[mywordindex,otherwordindex]),\
               distancemeans[mywordindex,otherwordindex],\
               numpy.sqrt(distancevariances[mywordindex,otherwordindex]),\
               absdistancepvalues[mywordindex,otherwordindex]))


# ----------------------------------------------------------



import re
# Define the sentence
sentence='For 20 years I have worked for 50 others, and I am sick of it, and for this reason, from today onward I work only for 1 person, myself.'
# Define and compile the pattern
pattern='[Ff]or [0-9]'
pattern=re.compile(pattern)
# Find all matches and print each match and its span
# (start and end character indices of the match) 
allmatches=re.finditer(pattern,sentence)
for tempmatch in allmatches:
    print(tempmatch.group(),tempmatch.span())



# ----------------------------------------------------------



import re
# Define the sentence
sentence='I want to go somewhere where the usual things I see everyday are nowhere to be seen.'
# Define and compile the pattern
pattern='\\b[Ww]here\\b'
pattern=re.compile(pattern)
# Find all matches and print each match and its span
# (start and end character indices of the match) 
allmatches=re.finditer(pattern,sentence)
for tempmatch in allmatches:
    print(tempmatch.group(),tempmatch.span())



# ----------------------------------------------------------