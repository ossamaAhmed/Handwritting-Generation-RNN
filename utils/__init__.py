# import numpy
# import matplotlib
# matplotlib.use('agg')
#
#
# def plot_stroke(stroke, save_name=None):
#     # Plot a single example.
#     f, ax = matplotlib.pyplot.subplots()
#
#     x = numpy.cumsum(stroke[:, 1])
#     y = numpy.cumsum(stroke[:, 2])
#
#     size_x = x.max() - x.min() + 1.
#     size_y = y.max() - y.min() + 1.
#
#     f.set_size_inches(5. * size_x / size_y, 5.)
#
#     cuts = numpy.where(stroke[:, 0] == 1)[0]
#     start = 0
#
#     for cut_value in cuts:
#         ax.plot(x[start:cut_value], y[start:cut_value],
#                 'k-', linewidth=3)
#         start = cut_value + 1
#     ax.axis('equal')
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#
#     if save_name is None:
#         matplotlib.pyplot.show()
#     else:
#         try:
#             matplotlib.pyplot.savefig(
#                 save_name,
#                 bbox_inches='tight',
#                 pad_inches=0.5)
#         except Exception:
#             print("Error building image!: " + save_name)
#
#         matplotlib.pyplot.close()
