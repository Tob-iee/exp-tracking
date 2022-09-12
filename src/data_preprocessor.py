# import the necessary packages
import os
import wget
import argparse
from zipfile import ZipFile
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.get_logger().setLevel('INFO')


# Initialize the parser
parser = argparse.ArgumentParser(description="Data Preprocessor")

# Add the parameters positional/optional
parser.add_argument("-dp",
                     dest= "datapath",
                     help="path directory to the raw images and csv",
                     default="./data_store/data/American Sign Language Letters.v1-v1.tensorflow/",
                     type=str)
parser.add_argument("-s",
                    dest= "datasplit",
                    nargs='+',
                    required=True,
                    help="the particular data split to be preprocessed (choose between train, valid, or test but if all enter all",
                    default="train test valid",
                    type=str)
parser.add_argument("-n",
                    dest= "tfrec",
                    help="the name for the tfrecord files",
                    default="Letters",
                    type=str)

parser.add_argument("-o",
                    dest= "output_path",
                    help="output path for the tfrecord files",
                    required=True,
                    default="./data_store/data_x/American Sign Language Letters.v1-v1.tfrecord",
                    type=str)

# parse the arguments
args = parser.parse_args()


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, example):
    feature = {
        "image/encoded": image_feature(image),
        "image/filename": bytes_feature(example["filename"]),
        "image/format": bytes_feature("jpg"),
        "image/height":  int64_feature(example["height"]),
        "image/object/bbox/xmin": float_feature(example["xmin"]),
        "image/object/bbox/ymin": float_feature(example["ymin"]),
        "image/object/bbox/xmax": float_feature(example["xmax"]),
        "image/object/bbox/ymax": float_feature(example["ymax"]),
        "image/object/class/label": int64_feature(example["label"]),
        "image/object/class/text": bytes_feature(example["class"]),
        "image/width": int64_feature(example["width"])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def tf_rec_writer(dataset_, path, data_split, recname, output_path):
    record_file = f"{recname}.tfrecords"
    output_dest =os.path.join(output_path, data_split)
    # output_dest = os.path.dirname(f"{out_path_join}")
    if not os.path.exists(output_dest):
        # Create output directory because it does not exist
        os.makedirs(output_dest)
    with tf.io.TFRecordWriter(f"{output_dest}/{record_file}") as writer:

        for data in dataset_:

            image = os.path.join(path, data['filename'])
            image = tf.io.decode_jpeg(tf.io.read_file(image))

            feature = create_example(image, data)
            writer.write(feature.SerializeToString())


def file_loader(FILE_PATH):
    count=0
    images = []
    annotations = []
    for files in os.listdir(FILE_PATH):

        if "csv" in files:
            annotations.append(files)
        else:
            images.append(files)

    return annotations, images


def convert_to_dict(FILE_PATH, annotations):
    dataset = pd.read_csv(os.path.join(FILE_PATH, annotations[0]))

    lb_make = LabelEncoder()
    dataset['label'] = lb_make.fit_transform(dataset[["class"]])

    dataset = dataset.to_dict(orient='records')
    return dataset


def data_source(url):
    # Define the remote file to retrieve
    remote_url = url
    # Define the local filename to save data
    local_dir = './data_store/data/American Sign Language Letters.v1-v1.tensorflow'
    # Make http request for remote file data
    file = wget.download(remote_url)
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(f'{file}', 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall(local_dir)

    os.remove(file)




def main():

    url ='https://public.roboflow.com/ds/NCbDlihA4z?key=pcdffiYbus'
    data_source(url)

    for splits in args.datasplit:

        FILE_PATH = args.datapath + splits

        annotations_test, images = file_loader(FILE_PATH)
        dataset_test = convert_to_dict(FILE_PATH, annotations_test)
        print(len(dataset_test))

        test_tfrec = tf_rec_writer(dataset_test, FILE_PATH, splits, "letters", args.output_path)


if __name__ ==  '__main__':
    main()
    # python src/data_preprocessor.py -s train test valid -o "./data_store/data_x/American Sign Language Letters.v1-v1.tfrecord"

