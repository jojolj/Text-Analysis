# -*- coding: utf-8 -*-

#%% Create TF-IDF vectors
n_docs=len(mycrawled_prunedtexts)
n_vocab=len(remainingvocabulary)
# Matrix of term frequencies
tfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
# Row vector of document frequencies
dfvector=scipy.sparse.lil_matrix((1,n_vocab))
# Loop over documents
for k in range(n_docs):
    # Row vector of which words occurred in this document
    temp_dfvector=scipy.sparse.lil_matrix((1,n_vocab))
    # Loop over words
    for l in range(len(mycrawled_prunedtexts[k])):
        # Add current word to term-frequency count and document-count
        currentword=myindices_in_prunedvocabulary[k][l]
        tfmatrix[k,currentword]=tfmatrix[k,currentword]+1
        temp_dfvector[0,currentword]=1
    # Add which words occurred in this document to overall document counts
    dfvector=dfvector+temp_dfvector
# Use the count statistics to compute the tf-idf matrix
tfidfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
# Let's use raw term count, and smoothed logarithmic idf
idfvector=numpy.squeeze(numpy.array(dfvector.todense()))
idfvector=1+numpy.log(((idfvector+1)**-1)*n_docs)
for k in range(n_docs):
    # Find nonzero term frequencies
    tempindices=numpy.nonzero(tfmatrix[k,:])[1]
    tfterm=numpy.squeeze(numpy.array(tfmatrix[k,tempindices].todense()))
    # Combine the tf and idf terms
    tfidfmatrix[k,tempindices]=tfterm*idfvector[tempindices]


#-----------------------



# Exclude header lines from each message
excludedlinemarkers=['Xref:','Path:','From:','Newsgroups:','Subject:','Summary:', \
    'Keywords:','Message-ID:','Date:','Expires:','Followup-To:','Distribution:', \
    'Organization:','Approved:','Supersedes:','Lines:','NNTP-Posting-Host:', \
    'References:','Sender:','In-Reply-To:','Article-I.D.:','Reply-To:', \
    'Nntp-Posting-Host:']
for k in range(len(mycrawled_texts)):
    print(k)
    templines=mycrawled_texts[k].splitlines()
    remaininglines=[]
    for l in range(len(templines)):
        line_should_be_excluded=0
        for m in range(len(excludedlinemarkers)):
            if len(templines[l])>=len(excludedlinemarkers[m]):
                if excludedlinemarkers[m]==\
                    templines[l][0:len(excludedlinemarkers[m])]:
                    line_should_be_excluded=1
                    break
        if line_should_be_excluded==0:
            remaininglines.append(templines[l])
    mycrawled_texts[k]='\n'.join(remaininglines)



#-----------------------



# Reduce the data to 500 highest-total TF-IDF features
dimensiontotals=numpy.squeeze(numpy.array( \
    numpy.sum(tfidfmatrix,axis=0)))
highesttotals=numpy.argsort(-1*dimensiontotals)
Xsmall=tfidfmatrix[:,highesttotals[0:500]]
Xsmall=Xsmall.todense()
# Normalize the documents to unit vector norm
tempnorms=numpy.squeeze(numpy.array(numpy.sum(numpy.multiply(Xsmall,Xsmall),axis=1)))
# If any documents have zero norm, avoid dividing them by zero
tempnorms[tempnorms==0]=1
Xsmall=scipy.sparse.diags(tempnorms**-0.5).dot(Xsmall)

import sklearn    
import sklearn.mixture
# Create the mixture model object, and
# choose the number of components and EM iterations
mixturemodel=sklearn.mixture.GaussianMixture(n_components=20, \
    covariance_type='diag',max_iter=100,init_params='random')

fittedmixture=mixturemodel.fit(Xsmall)
sklearn_mixturemodel_means=fittedmixture.means_
sklearn_mixturemodel_weights=fittedmixture.weights_
sklearn_mixturemodel_covariances=fittedmixture.covariances_



#-----------------------



# Find top 20 words with highest mean feature value for each cluster
for k in range(n_components):
    print(k)
    highest_dimensionweight_indices=numpy.argsort( \
        -numpy.squeeze(sklearn_mixturemodel_means[k,:]),axis=0)
    highest_dimensionweight_indices=highesttotals[highest_dimensionweight_indices]
    print(' '.join(remainingvocabulary[highest_dimensionweight_indices[1:20]]))



#-----------------------



#%% Use the TF-IDF matrix as data to be clustered
X=tfidfmatrix
# Normalize the documents to unit vector norm
tempnorms=numpy.squeeze(numpy.array(numpy.sum(X.multiply(X),axis=1)))
# If any documents have zero norm, avoid dividing them by zero
tempnorms[tempnorms==0]=1
X=scipy.sparse.diags(tempnorms**-0.5).dot(X)

n_data=numpy.shape(X)[0]
n_dimensions=numpy.shape(X)[1]



#-----------------------



#%% Initialize the Gaussian mixture model

# Function to initialize the Gaussian mixture model, create component parameters
def initialize_mixturemodel(X,n_components):
    # Create lists of sparse matrices to hold the parameters
    n_dimensions=numpy.shape(X)[1]
    mixturemodel_means=scipy.sparse.lil_matrix((n_components,n_dimensions))
    mixturemodel_weights=numpy.zeros((n_components))
    mixturemodel_covariances=[]
    mixturemodel_inversecovariances=[]
    for k in range(n_components):
        tempcovariance=scipy.sparse.lil_matrix((n_dimensions,n_dimensions))    
        mixturemodel_covariances.append(tempcovariance)
        tempinvcovariance=scipy.sparse.lil_matrix((n_dimensions,n_dimensions))   
        mixturemodel_inversecovariances.append(tempinvcovariance)

    # Initialize the parameters
    for k in range(n_components):
        mixturemodel_weights[k]=1/n_components
        # Pick a random data point as the initial mean
        tempindex=scipy.stats.randint.rvs(low=0,high=n_components)
        mixturemodel_means[k]=X[tempindex,:].toarray()  
        # Initialize the covariance matrix to be spherical
        for l in range(n_dimensions):
            mixturemodel_covariances[k][l,l]=1
            mixturemodel_inversecovariances[k][l,l]=1
    return(mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\ 
           mixturemodel_inversecovariances)



#-----------------------




def run_estep(X,mixturemodel_means,mixturemodel_covariances, \  
              mixturemodel_inversecovariances,mixturemodel_weights):
    # For each component, compute terms that do not involve data
    meanterms=numpy.zeros((n_components))
    logdeterminants=numpy.zeros((n_components))
    logconstantterms=numpy.zeros((n_components))
    for k in range(n_components):
        # Compute mu_k*inv(Sigma_k)*mu_k
        meanterms[k]=(mixturemodel_means[k,:]* \
            mixturemodel_inversecovariances[k]*mixturemodel_means[k,:].T)[0,0]
        # Compute determinant of Sigma_k. For a diagonal matrix 
        # this is just the product of the main diagonal
        logdeterminants[k]=numpy.sum(numpy.log(mixturemodel_covariances[k].diagonal(0)))
        # Compute constant term beta_k * 1/(|Sigma_k|^1/2)
        # Omit the (2pi)^d/2 as it cancels out
        logconstantterms[k]=numpy.log(mixturemodel_weights[k]) - 0.5*logdeterminants[k]

    print('E-step part2 ')
    # Compute terms that involve distances of data from components
    xnorms=numpy.zeros((n_data,n_components))
    xtimesmu=numpy.zeros((n_data,n_components))
    for k in range(n_components):
        print(k)
        xnorms[:,k]=(X*mixturemodel_inversecovariances[k]*X.T).diagonal(0)
        xtimesmu[:,k]=numpy.squeeze((X*mixturemodel_inversecovariances[k]* \
            mixturemodel_means[k,:].T).toarray())
    xdists=xnorms+numpy.matlib.repmat(meanterms,n_data,1)-2*xtimesmu        
    # Substract maximal term before exponent (cancels out) to maintain computational precision
    numeratorterms=logconstantterms-xdists/2
    numeratorterms-=numpy.matlib.repmat(numpy.max(numeratorterms,axis=1),n_components,1).T
    numeratorterms=numpy.exp(numeratorterms)
    mixturemodel_componentmemberships=numeratorterms/numpy.matlib.repmat( \
        numpy.sum(numeratorterms,axis=1),n_components,1).T
    return(mixturemodel_componentmemberships)



#-----------------------



def run_mstep_sumweights(mixturemodel_componentmemberships):
    # Compute total weight per component
    mixturemodel_weights=numpy.sum(mixturemodel_componentmemberships,axis=0)
    return(mixturemodel_weights)

def run_mstep_means(X,mixturemodel_componentmemberships,mixturemodel_weights):
    # Update component means
    mixturemodel_means=scipy.sparse.lil_matrix((n_components,n_dimensions))
    for k in range(n_components):
        mixturemodel_means[k,:]=\
            numpy.sum(scipy.sparse.diags(mixturemodel_componentmemberships[:,k]).dot(X),axis=0)
        mixturemodel_means[k,:]/=mixturemodel_weights[k]
    return(mixturemodel_means)

def run_mstep_covariances(X,mixturemodel_componentmemberships,mixturemodel_weights,mixturemodel_means):
    # Update diagonal component covariance matrices
    n_dimensions=numpy.shape(X)[1]
    n_components=numpy.shape(mixturemodel_componentmemberships)[1]
    tempcovariances=numpy.zeros((n_components,n_dimensions))
    mixturemodel_covariances=[]
    mixturemodel_inversecovariances=[]
    for k in range(n_components):
        tempcovariances[k,:]= \
            numpy.sum(scipy.sparse.diags(mixturemodel_componentmemberships[:,k]).dot(X.multiply(X)),axis=0) \
            -mixturemodel_means[k,:].multiply(mixturemodel_means[k,:])*mixturemodel_weights[k]
        tempcovariances[k,:]/=mixturemodel_weights[k]
        # Convert to sparse matrices
        tempepsilon=1e-10
        # Add a small regularization term
        temp_covariance=scipy.sparse.diags(tempcovariances[k,:]+tempepsilon)
        temp_inversecovariance=scipy.sparse.diags((tempcovariances[k,:]+tempepsilon)**-1)
        mixturemodel_covariances.append(temp_covariance)
        mixturemodel_inversecovariances.append(temp_inversecovariance) 
    return(mixturemodel_covariances,mixturemodel_inversecovariances)   

def run_mstep_normalizeweights(mixturemodel_weights):
    # Update mixture-component prior probabilities
    mixturemodel_weights/=sum(mixturemodel_weights)
    return(mixturemodel_weights)



#-----------------------



#%% Perform the EM algorithm iterations    
def perform_emalgorithm(X,n_components,n_emiterations):
    mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
        mixturemodel_inversecovariances=initialize_mixturemodel(X,n_components)

    for t in range(n_emiterations):
        # ====== E-step: Compute the component membership
        # probabilities of each data point ======
        print('E-step ' + str(t))
        mixturemodel_componentmemberships=run_estep(X,mixturemodel_means,mixturemodel_covariances,\
            mixturemodel_inversecovariances,mixturemodel_weights)    
        # ====== M-step: update component parameters======
        print('M-step ' + str(t))
        print('M-step part1 ' + str(t))
        mixturemodel_weights=run_mstep_sumweights(mixturemodel_componentmemberships)
        print('M-step part2 ' + str(t))
        mixturemodel_means=run_mstep_means(X,mixturemodel_componentmemberships,mixturemodel_weights)
        print('M-step part3 ' + str(t))
        mixturemodel_covariances,mixturemodel_inversecovariances=run_mstep_covariances(X,\
            mixturemodel_componentmemberships,mixturemodel_weights,mixturemodel_means)
        print('M-step part4 ' + str(t))
        mixturemodel_weights=run_mstep_normalizeweights(mixturemodel_weights)
    return(mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
        mixturemodel_inversecovariances)
# Try out the functions we just defined on the data
n_components=20
n_emiterations=100
mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\ 
        mixturemodel_inversecovariances = perform_emalgorithm(X,n_components,n_emiterations)



#-----------------------



# Find top 20 words for each cluster
for k in range(n_components):
    print(k)
    highest_dimensionweight_indices=\
        numpy.argsort(-numpy.squeeze(\
        mixturemodel_means[k,:].toarray()),axis=0)       
    print(''.join(remainingvocabulary[\
        highest_dimensionweight_indices[1:20]]))



#-----------------------



# Version 1 - Get component membership probabilities for each document d
# Find the document d with highest-probability p(k|d) to be from cluster k
for k in range(n_components):
    tempprobs=numpy.array(numpy.squeeze(mixturemodel_componentmemberships[:,k]))
    highest_componentprob_indices=numpy.argsort(-tempprobs,axis=0)
    print(k)
    print(highest_componentprob_indices[0:10])
    print(' '.join(mycrawled_nltktexts[highest_componentprob_indices[0]]))

# Version 2 - Get documents closest to component mean, i.e. highest p(d|k).
# ---The computation of distances here is the same as done in the E-step of EM---
# For each component, compute terms that do not involve data
meanterms=numpy.zeros((n_components))
logdeterminants=numpy.zeros((n_components))
logconstantterms=numpy.zeros((n_components))
for k in range(n_components):
    # Compute mu_k*inv(Sigma_k)*mu_k
    meanterms[k]=(mixturemodel_means[k,:]* \
        mixturemodel_inversecovariances[k]*mixturemodel_means[k,:].T)[0,0]
# Compute terms that involve distances of data from components
xnorms=numpy.zeros((n_data,n_components))
xtimesmu=numpy.zeros((n_data,n_components))
for k in range(n_components):
    xnorms[:,k]=(X*mixturemodel_inversecovariances[k]*X.T).diagonal(0)
    xtimesmu[:,k]=numpy.squeeze((X*mixturemodel_inversecovariances[k]* \
        mixturemodel_means[k,:].T).toarray())
xdists=xnorms+numpy.matlib.repmat(meanterms,n_data,1)-2*xtimesmu

for k in range(n_components):
    tempdists=numpy.array(numpy.squeeze(xdists[:,k]))
    highest_componentprob_indices=numpy.argsort(tempdists,axis=0)
    print(k)
    print(highest_componentprob_indices[0:10])
    print(' '.join(mycrawled_nltktexts[highest_componentprob_indices[0]]))



#-----------------------


