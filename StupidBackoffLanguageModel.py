import math, collections
class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    self.t = 0
    a = 'any_word'
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.bigramCounts[(a,token)] = self.bigramCounts[(a,token)] + 1
        a = token
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.t += 1
    self.total = len(self.unigramCounts)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    a = '<s>'  
    for i in range(1, len(sentence)):
      count = (self.bigramCounts[(a,sentence[i])]) + 1
      d = self.unigramCounts[a] + self.total
      if count > 1:
        score += math.log(count)
        score -= math.log(d)
      else:
        score += 0.4 * (math.log(self.unigramCounts[sentence[i]] + 1))
        #score -= math.log(1+ self.total)
        score -= math.log(self.t + self.total)        
      a = sentence[i]
    return score
