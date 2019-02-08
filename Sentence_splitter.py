from koalanlp import *
import tensorflow as tf
import datetime


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None,
    "The input datadir with path",
)

flags.DEFINE_string(
    "output_file", None,
    "output_file with path"
)

def main(_):
	"""
	python3 Sentence_splitter.py --input_file="../mini_news.utf8" --output_file="../mini_news_sent.utf8"
	"""
	initialize(packages=[API.HANNANUM], version="1.9.4", java_options="-Xmx4g")

	sent_split = SentenceSplitter(splitter_type=API.HANNANUM)

	file_write = open(FLAGS.output_file, 'w')

	with open(FLAGS.input_file) as f:
	    i=0
	    for line in f:
	        
	        if("###" in line):
	            continue 
	        if("2001 Joins.com" in line):
	            continue
	        
	     
	        sentences = sent_split.sentences(line)
	        if(len(sentences)==0):
	            continue
	        

	        for sent in sentences:
	            file_write.write(sent+'\n')
	        i+=1
	        if(i%100000==0):
	            print(datetime.datetime.now().isoformat())
	file_write.close()

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")

    tf.app.run()

