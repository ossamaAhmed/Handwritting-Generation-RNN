import tensorflow as tf


def add_names_for_inference(vars_dict):
    res = []
    for name, var in vars_dict.items():
        res.append(tf.identity(var, name=name))
    return res


def create_summaries(vars_dict):
    for name, var in vars_dict.items():
        tf.summary.scalar(name + '_minimum', tf.reduce_min(var))
        tf.summary.scalar(name + '_maximum', tf.reduce_max(var))
    return tf.summary.merge_all()
