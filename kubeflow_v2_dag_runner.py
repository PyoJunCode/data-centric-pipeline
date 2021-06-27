
"""Define KubeflowV2DagRunner to run the pipeline."""

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.proto import trainer_pb2
from tfx.tools.cli.kubeflow_v2 import labels

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
_OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to metadata service backend.
_PIPELINE_ROOT = os.path.join(_OUTPUT_DIR, 'tfx_pipeline_output',
                              configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
_SERVING_MODEL_DIR = os.path.join(_PIPELINE_ROOT, 'serving_model')

_DATA_PATH = 'gs://{}/data-centric/data/'.format(configs.GCS_BUCKET_NAME)


def run():
  """Define a pipeline to be executed using Kubeflow V2 runner."""

  tfx_image = os.environ.get(labels.TFX_IMAGE_ENV)
  project_id = os.environ.get(labels.GCP_PROJECT_ID_ENV)
  api_key = os.environ.get(labels.API_KEY_ENV)

  runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
      project_id=project_id,
      display_name='tfx-kubeflow-v2-pipeline-{}'.format(configs.PIPELINE_NAME),
      default_image=tfx_image)

  dsl_pipeline = pipeline.create_pipeline(
      pipeline_name=configs.PIPELINE_NAME,
      pipeline_root=_PIPELINE_ROOT,
      data_path=_DATA_PATH,
      preprocessing_fn=configs.PREPROCESSING_FN,
      run_fn=configs.RUN_FN,
      train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
      eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
      eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
      serving_model_dir=_SERVING_MODEL_DIR,

  )

  runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
      config=runner_config)

  if os.environ.get(labels.RUN_FLAG_ENV, False):
    # Only trigger the execution when invoked by 'run' command.
    runner.run(
        pipeline=dsl_pipeline, api_key=api_key)
  else:
    runner.compile(pipeline=dsl_pipeline, write_out=True)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
