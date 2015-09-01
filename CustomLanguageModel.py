import math, collections
class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
  
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    a = 'any_word'
    b = 'anywordd'
    c = 'anyworddd'
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.bigramCounts[(a,token)] = self.bigramCounts[(a,token)] + 1
        a = token
        self.bigramCounts[(b,c,token)] = self.bigramCounts[(b,c,token)] + 1
        b = c
        c = token
        self.unigramCounts[token] = self.unigramCounts[token] + 1
    self.total = len(self.unigramCounts)
    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    a = '<s>'  
    for i in range(1, len(sentence)-1):
      count = (self.trigramCounts[(a,sentence[i],sentence[i+1])]) + 1
      d = self.bigramCounts[(a,sentence[i])] + self.total     
      score += math.log(count)
      score -= math.log(d)        
      a = sentence[i]
    return score
