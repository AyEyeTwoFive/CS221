#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *
import collections

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    feat = collections.defaultdict(int)
    words = x.split()
    for w in words:
        feat[w] += 1
    return feat
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def predict(x):
        if dotProduct(weights, featureExtractor(x)) > 0:
            return 1
        else:
            return -1

    for x, y in trainExamples:
        for feat in featureExtractor(x):
            weights[feat]=0
    for i in range(numEpochs):
        for x,y in trainExamples:
            if dotProduct(weights,featureExtractor(x))*y < 1:
                increment(weights,eta*y,featureExtractor(x))
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {feat: random.random() for feat in random.sample(list(weights),len(weights)-1)}
        if dotProduct(weights,phi) > 0:
            y = 1
        else:
            y = -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        feat = collections.defaultdict(int)
        x = x.replace(' ','')
        for i in range(len(x)+1-n):
            feat[x[i:i+n]] += 1
        return feat
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:
# See sentiment.pdf


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    centroids = [sample.copy() for sample in random.sample(examples, K)]
    z = [random.randint(0, K - 1) for i in examples]
    d = [0 for item in examples]
    matched = None
    xsquared = []
    for item in examples:
        stored = collections.defaultdict(float)
        for k, v in item.items():
            stored[k] = v * v
        xsquared.append(stored)
    for r in range(maxEpochs):
        csquared = []
        for item in centroids:
            stored = collections.defaultdict(float)
            for k, v in item.items():
                stored[k] = v * v
            csquared.append(stored)
        for index, item in enumerate(examples):
            min_d = 99999
            for i, cluster in enumerate(centroids):
                dist = sum(xsquared[index].values()) + sum(csquared[i].values())
                for k in set(item.keys() & cluster.keys()):
                    dist -= 2 * item[k] * cluster[k]
                if dist < min_d:
                    min_d = dist
                    z[index] = i
                    d[index] = min_d
        if matched == z:
            break
        else:
            ncluster = [0 for cluster in centroids]
            for i, cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k] = 0.0
            for index, item in enumerate(examples):
                ncluster[z[index]] += 1
                cluster = centroids[z[index]]
                for k, v in item.items():
                    if k in cluster:
                        cluster[k] += v
                    else:
                        cluster[k] = 0.0 + v
            for i, cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k] /= ncluster[i]
            matched = z[:]
    return centroids, z, sum(d)
    # END_YOUR_CODE
