import os
import time
import shutil
import argparse
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
print(mlflow.__version__)

tf.get_logger().setLevel('INFO')
strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

# Initialize the parser
parser = argparse.ArgumentParser(description="Hand_sign_trainer")

# Add the parameters positional/optional
parser.add_argument("-dp",
                     dest= "datapath",
                     help="path directory to the training data",
                     default="./data_store/data/American Sign Language Letters.v1-v1.tfrecord/",
                     type=str)
parser.add_argument("-s",
                    dest= "datasplit",
                    nargs='+',
                    help="the particular data split to be trained on(enter a set of split as training and validation set",
                    default=["train", "valid"],
                    type=list)
parser.add_argument("-n",
                    dest= "tfrec",
                    help="the name for the tfrecord files",
                    default="letters",
                    type=str)
parser.add_argument("-ap",
                    dest= "arti",
                    help="set location to store model artifact",
                    default="./data_store/artifacts",
                    type=str)
parser.add_argument("-en",
                    dest= "exp_name",
                    help="defines the experiment name to start",
                    default="Hand_Signs_Exp1",
                    type=str)
parser.add_argument("-dt",
                    dest= "dagshub_train",
                    help="to specify if thet training with be run on dagshub",
                    required=True,
                    default="False",
                    type=bool)

# parse the arguments
args = parser.parse_args()

URI = os.environ.get('MLFLOW_TRACKING_URI')

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

FILENAMES_PATH = args.datapath
EXPERIMENT_NAME = args.exp_name
LOCAL_ARTIFACTS_PATH = args.arti
DAGSHUB_TRAINING = args.dagshub_train

TRAINING_FILENAMES =  FILENAMES_PATH + args.datasplit[0] + "/letters.tfrecords"
VALID_FILENAMES = FILENAMES_PATH + args.datasplit[1] + "/letters.tfrecords"

print("Train TFRecord Files:", TRAINING_FILENAMES)
print("Validation TFRecord Files:", VALID_FILENAMES)



# Create the dataset object for tfrecord file(s)

def load_dataset(tf_filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False  # disable order, increase speed
  dataset = tf.data.TFRecordDataset(tf_filenames)

  dataset = dataset.with_options(ignore_order)

  return dataset

# Decoding function
def parse_record(record):

  tfrecord_feat_format = (
              {
                  "image/encoded": tf.io.FixedLenFeature([], tf.string),
                  "image/filename": tf.io.FixedLenFeature([], tf.string),
                  "image/format": tf.io.FixedLenFeature([], tf.string),
                  "image/height": tf.io.FixedLenFeature([], tf.int64),
                  "image/object/bbox/xmax": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/xmin": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/ymax": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/ymin": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
                  "image/object/class/text": tf.io.FixedLenFeature([], tf.string),
                  "image/width": tf.io.FixedLenFeature([], tf.int64),
              }
          )



  example = tf.io.parse_single_example(record, tfrecord_feat_format)



  IMAGE_SIZE = [400, 400]

  image =  tf.io.decode_jpeg(example["image/encoded"], channels=3)
  image = tf.cast(image, tf.float32)

  xmax = tf.cast(example["image/object/bbox/xmax"], tf.int32)
  xmin = tf.cast(example["image/object/bbox/xmin"], tf.int32)
  ymax = tf.cast(example["image/object/bbox/ymax"], tf.int32)
  ymin = tf.cast(example["image/object/bbox/ymin"], tf.int32)

  box_width = xmax - xmin
  box_height = ymax - ymin
  image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, IMAGE_SIZE)

# more feature preprocessing
  image = tf.image.random_flip_left_right(image)
  # image = tfa.image.rotate(image, 40, interpolation='NEAREST')


  # image = tf.cast(image, "uint8")
  # image = tf.image.encode_jpeg(image, format='rgb', quality=100)


  label = example["image/object/class/label"]
  label = tf.cast(label, tf.int32)
  # label = tf.one_hot(label, depth=26)


  return (image, label)

def get_dataset(filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False  # disable order, increase speed
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.with_options(ignore_order)

  dataset = dataset.map(parse_record, num_parallel_calls=AUTOTUNE)
  dataset = dataset.cache()

  dataset = dataset.shuffle(buffer_size=10 * BATCH_SIZE, reshuffle_each_iteration=True)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  # dataset = dataset.repeat()

  return dataset


def get_cnn():
  model = tf.keras.Sequential([

  tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu', input_shape=[400, 400, 3]),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(26,'softmax')
  ])

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )

  return model


def main():

  mlflow.set_tracking_uri(URI)

  tfr_dataset = get_dataset(TRAINING_FILENAMES)
  print(tfr_dataset)

  tfr_testdata = get_dataset(VALID_FILENAMES)
  print(tfr_testdata)

  # if not os.path.exists(args.arti):
    # Create artifacts directory because it does not exist
    # os.makedirs(args.arti)

  print(f"The tracking uri is: {mlflow.get_tracking_uri()}")

  client = MlflowClient()
  client_list = client.list_experiments()
  search_exp = client.get_experiment_by_name(EXPERIMENT_NAME)
  print(client_list)

  search_exp = None
  for experiment in client_list:
    schema = dict(experiment)
    if schema["name"] == EXPERIMENT_NAME:
      print(schema["name"])
      search_exp == True
      pass


  if client_list == None or search_exp != True:
    # create and set experiment
    experiment_new = mlflow.create_experiment(EXPERIMENT_NAME)
    client.set_experiment_tag(experiment_new, "CV.framework", "Tensorflow_CV")
    # experiment = client.get_experiment(experiment_new)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

  elif search_exp == True:
    # Set experiment
    # mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

  else:
    print("Please check your experiment name it might have been deleted")

  artifact_uri = mlflow.get_artifact_uri()
  print(f"The artifacts uri is: {artifact_uri}")

  model = get_cnn()
  model.summary()

  mlflow.tensorflow.autolog(every_n_iter=2)

  # start experiment tracking runs
  with mlflow.start_run(experiment_id=experiment.experiment_id):

    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")

    # Training
    start_training = time.time()
    history = model.fit(tfr_dataset,
              epochs=2, verbose=1)
    end_training = time.time()

    training_time = end_training - start_training

    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_metric('batchsize', BATCH_SIZE)
    mlflow.log_metric('training_accuracy', history.history['sparse_categorical_accuracy'][-1])
    mlflow.log_metric('training_loss', history.history['loss'][-1])
    mlflow.log_metric('training_time', training_time)

    tfr_testdata = get_dataset(VALID_FILENAMES)

    start_evaluating = time.time()
    val_loss, val_accuracy = model.evaluate(tfr_testdata)

    end_evaluating = time.time()
    evaluating_time = end_evaluating - start_evaluating


    mlflow.log_metric('validation_accuracy', val_accuracy)
    mlflow.log_metric('validation_loss', val_loss)
    mlflow.log_metric('evaluating_time', evaluating_time)

    run = mlflow.get_run(run.info.run_id)
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    print("--")
    mlflow.end_run()

  # Check for any active runs
  print(f"Active run: {mlflow.active_run()}")


if __name__ ==  '__main__':
  main()