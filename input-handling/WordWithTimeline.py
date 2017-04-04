class WordWithTimeline:

	def __init__(self, word, start, end):
		self.word = word
		self.start = start
		self.end = end

	def getWord(self):
		return self.word

	def getStartTime(self):
		return self.start

	def getEndTime(self):
		return self.end