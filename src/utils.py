import argparse
import numpy as np
import tensorflow as tf
import scipy.misc as smc
from io import BytesIO


def is_prcnt(x):
    '''Assess whether or not the value is a float between 0 and 1'''
    error_msg = 'value must be a float in range [0, 1)'
    try:
        x = float(x)
    except:
        raise argparse.ArgumentTypeError(error_msg)
    if x >= 0 and x < 1:
        return x
    raise argparse.ArgumentTypeError(error_msg)


class Logger(object):
    '''
    TensorBoard logger for pytorch
    '''
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            smc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i),
                                                  image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def write_to_logs(logger, model, info, step, histo=None):
    # (1) Log the scalar values
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)

    # (2) Log values and gradients of the parameters (histogram)
    if histo:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag+'/grad', to_np(value.grad), step)

## Instantiate and Call
logger = Logger(os.path.join(logs_path, model_name))
write_to_logs(logger, model, info, step, histo=histo)
