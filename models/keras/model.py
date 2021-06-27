
import tensorflow as tf
keras = tf.keras
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import schema_pb2

LABEL_KEY = 'label'
DROP_FEATURES = []
NUM_CLASSES = 10

def transformed_name(name):
  return name + '_xf'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


@tf.function
def decode_and_resize(image):
    return tf.image.resize(tf.io.decode_png(image), (256, 256))


@tf.function
def parse_png_images(png_images):
  with tf.device("/cpu:0"):
    flattened = tf.reshape(png_images, [-1])
    decoded = tf.map_fn(decode_and_resize, flattened, dtype=tf.float32)
    reshaped = tf.reshape(decoded, [-1, 256, 256, 3])
    return reshaped / 255.


def _build_estimator(config, num_filters=None):

  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=[1], dtype="string", name="image_xf"))
  model.add(keras.layers.Lambda(parse_png_images))
  for filters in num_filters:
      model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, activation="relu"))
      model.add(keras.layers.MaxPool2D())
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
  model.compile(loss="sparse_categorical_crossentropy",
                optimizer="adam", metrics=["accuracy"])
  
  return model

def _example_serving_receiver_fn(tf_transform_output, schema):

  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)
  for feature in DROP_FEATURES + [LABEL_KEY]:
    transformed_features.pop(transformed_name(feature))

  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):

  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)


  transformed_features = tf_transform_output.transform_raw_features(
      features)

  # The key name MUST be 'examples'.
  receiver_tensors = {'examples': serialized_tf_example}

  features.update(transformed_features)
  for feature in DROP_FEATURES + [LABEL_KEY]:
    if feature in features:
        features.pop(feature)
    if transformed_name(feature) in features:
        features.pop(transformed_name(feature))
  features.pop('image')
  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=transformed_features[transformed_name(LABEL_KEY)])


def _input_fn(filenames, tf_transform_output, batch_size=200):

  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  transformed_features = dataset.make_one_shot_iterator().get_next()

  for feature in DROP_FEATURES:
    transformed_features.pop(transformed_name(feature))

  return transformed_features, transformed_features.pop(
      transformed_name(LABEL_KEY))


def run_fn(hparams):


  schema = io_utils.parse_pbtxt_file(hparams.schema_file, schema_pb2.Schema())

  train_batch_size = 40
  eval_batch_size = 40
  num_cnn_layers = 4
  first_cnn_filters = 32

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  train_input_fn = lambda: _input_fn(
      hparams.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(
      hparams.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('uc-merced', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='uc-merced-val')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  num_filters = [first_cnn_filters]
  for layer_index in range(1, num_cnn_layers):
    num_filters.append(num_filters[-1] * 2)

  estimator = _build_estimator(
      config=run_config,
      num_filters=num_filters)
      


  estimator.save(hparams.serving_model_dir)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }