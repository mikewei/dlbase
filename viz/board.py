import matplotlib.pyplot as plt
from dlbase.utils.py import save_params
from dlbase.utils.nb import FigureDisplay

class ProgressBoard(FigureDisplay):
    """The board that plots data points in animation."""

    @save_params
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':']*2, colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                 fig=None, axes=None, figsize=(8, 2.5), is_display=True):
        super().__init__(hint='<ProgressBoard>')

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        import collections
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) < every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.is_display:
            return
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                        linestyle=ls, color=color)[0])
            labels.append(k + f'({v[-1].y:.2f})' if len(v) > 0 else '')
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        self.show()