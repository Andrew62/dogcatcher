
import tensorflow as tf
from inception_preprocessing import preprocess_image

def batch_producer(filepath, n_classes, **kwargs):
    """Function for loading batches of images and
    and labels from a csv *without* a header. CSV files
    must be in the format of
        class_code,/abs/path/to/img
        class_code,/abs/path/to/img
        class_code,/abs/path/to/img

    Parameters
    -----------
    filepath : list
        list of paths to csv files. Even if just using one file, it must
        be a list. For example ['/path/to/file.csv']
    n_classes : int
        number of classes to be used in one-hot encoding
    batch_size : (kwarg) int
        number of samples per batch. Default is 4
    epochs : (kwarg) int
        number of epochs to run. Default is 70
    img_shape : (kwarg) tuple
        shape of the image. Must be in the form of (H,W,C). Image
        will *not* be resized, the value is used for setting
        the shape for the batch queue. Default is (224, 224, 3)
    is_training : (kwarg) bool
        when set to true, the loader will apply image transformations.
        Default is True
    num_threads : (kwarg) int
        number of threads to use for the loader. Default is 4
    """
    batch_size = kwargs.pop("batch_size", 4)
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    num_threads = kwargs.pop("num_threads", 4)
    epochs = kwargs.pop("epochs", 70)
    is_training = kwargs.pop("is_trianing", True)

    # loads a series of text files
    filename_queue = tf.train.string_input_producer(filepath, num_epochs=epochs)

    # used to read each text file line by line
    reader = tf.TextLineReader()

    # actually parse the text file. returns idx, content
    _, record = reader.read(filename_queue)

    # split out the csv. Defaults to returning strings.
    img_class, fname = tf.decode_csv(record, record_defaults=[[1], [""]])

    # read the image file
    content = tf.read_file(fname)

    # decode buffer as jpeg
    img_raw = tf.image.decode_jpeg(content, channels=img_shape[-1])

    img_content = preprocess_image(img_raw, img_shape[0], img_shape[1], is_training=is_training)

    # setting the shape is neccessary for the shuffle_batch op. Fails otherwise
    # img_content.set_shape(img_shape)

    # load batches of images all multithreaded like
    class_batch, img_batch = tf.train.shuffle_batch([img_class, img_content],
                                                    batch_size=batch_size,
                                                    capacity=batch_size * 4,
                                                    num_threads=num_threads,
                                                    min_after_dequeue=batch_size * 2)

    one_hot_classes = tf.one_hot(class_batch, depth=n_classes,
                                 on_value=1.0, off_value=0.0)
    return one_hot_classes, img_batch
