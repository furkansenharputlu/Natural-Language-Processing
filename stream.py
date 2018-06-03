# Imports/general tools
import time

# Imports/general tools/encoding stdout
import sys
import codecs

# Import JSON parsing tools
import json

# Imports/tweepy specific
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

# Credentials
consumer_key = 'lWnCDeBKmezIP4JNVtSsyo32o'
consumer_secret = 'TLacVvt5AycJ6reXrJJnwrG7u2LOSht7dN3mea9CpPAnSL8joZ'
access_token = '974266007249477632-VMTBGUd9chNgeJFj6VfEmeC8MZAwYSH'
access_secret = '68ug1mD3DWyto98iHRyEJXq4HwKXYh9f1Xp8P0bnozUO5'

# To prevent encoding errors (charmap codec errors, etc) in the stdout stream


# Prompt for user input
filename = 'recep'

# Twizzer stream class
class listener(StreamListener):

	def on_data(self, data):
		try:
			
			jsonData = json.loads(data)
			
			# alternative approach using json library (suggested by Satish Chandra)
			createdAt = jsonData['created_at']
			text = jsonData['text']
			text=text.replace("\n"," ")


			# concatenate the timestamp, an arbitrary separator and the text of the tweet
			save_this = createdAt+'=>'+text

			# print to stdout
			print (save_this)

			# open file for writing, in append mode so that updates don't erase previous work
			save_file = open(filename+'.csv', 'a')
			
			# set file encoding to utf-8 and write to file
			save_file.write(save_this)
			save_file.write('\n')
			save_file.close()
			return True
		except BaseException as e:
			print ('failed ondata, ', str(e))
			time.sleep(5)

	def on_error(self, status):
		print (status)

# The meat of the script, authentication first, then streaming
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

stream = Stream(auth, listener())

# Set the search terms here! As far as I
# can tell, adding multiple terms inside:
#        track=['term1', 'term2', 'term_n']
# returns the results of a Boolean OR
stream.filter(track=['recep'])