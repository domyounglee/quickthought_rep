import tensorflow as tf
import data_utils 
import collections

SentenceBatch = collections.namedtuple("Sentence_batch", ("ids", "mask"))

def prefetch_input_data(file_pattern,batch_size=32,epochs=1):
	"""parse a batch of tf.Example protos."""
	def _sparse_to_batch(sparse):
		ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes. which are shorter than the max length instance
		mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
								tf.ones_like(sparse.values, dtype=tf.int32))#tensor = tf.constant([[1, 2, 3], [4, 5, 6]]) -> # [[1, 1, 1], [1, 1, 1]]
		return SentenceBatch(ids=ids, mask=mask)

	data_files = []
	for pattern in file_pattern.split(","):
		data_files.extend(tf.gfile.Glob(pattern))
	if not data_files:
		tf.logging.fatal("Found no input files matching %s", file_pattern)
	else:
		tf.logging.info("Prefetching values from %d files matching %s",
						len(data_files), file_pattern)


	dataset=tf.data.TFRecordDataset(data_files)

	dataset=dataset.map(data_utils.parse_function)
	dataset = dataset.repeat(epochs)
	dataset = dataset.batch(batch_size)

	#iterator=dataset.make_initializable_iterator()
	iterator=dataset.make_one_shot_iterator()


	next_element=iterator.get_next()
	return _sparse_to_batch(next_element)

























def main(_):


	filenames =  ['./TFRecord/train-00000-of-00100', './TFRecord/train-00001-of-00100']

	dataset=tf.data.TFRecordDataset(filenames)

	dataset=dataset.map(data_utils.parse_function)
	dataset = dataset.repeat()
	dataset = dataset.batch(3)

	#iterator=dataset.make_initializable_iterator()
	iterator=dataset.make_one_shot_iterator()


	next_element=iterator.get_next()
	tf.sparse_tensor_to_dense(next_element)
	tf.sparse_to_dense(next_element.indices,next_element.dense_shape,
						tf.ones_like(next_element.values,dtype=tf.int32))
	with tf.Session() as sess:
		#sess.run(iterator.initializer, feed_dict={filenames:train_filenames})
		

		print(sess.run(next_element))
		print(sess.run(tf.sparse_tensor_to_dense(next_element)))
		print(sess.run(tf.sparse_to_dense(next_element.indices,next_element.dense_shape,tf.ones_like(next_element.values,dtype=tf.int32))))

if __name__ == "__main__":
  tf.app.run()

