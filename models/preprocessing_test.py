
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import preprocessing


class PreprocessingTest(tf.test.TestCase):

  def testPreprocessingFn(self):
    self.assertTrue(callable(preprocessing.preprocessing_fn))


if __name__ == '__main__':
  tf.test.main()
