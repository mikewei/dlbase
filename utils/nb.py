from typing import Literal, Iterable
from collections import deque, namedtuple
from contextlib import AbstractContextManager
from IPython.display import display
from .py import NopContextManager

def new_display(hint=None):
    return display(display_id=True) if hint is None else display(hint, display_id=True)

class TextDisplay(NopContextManager):
    def __init__(self, hint='<TextDisplay>'):
        self.display = new_display(hint)
    def update(self, text, format: Literal['default', 'markdown', 'html'] = 'default'):
        if format == 'default':
            self.display.update(text)
        elif format == 'markdown':
            from IPython.display import Markdown
            self.display.update(Markdown(text))
        elif format == 'html':
            from IPython.display import HTML
            self.display.update(HTML(text))

class TailDisplay(TextDisplay):
    def __init__(self, last_n=1, format: Literal['default', 'markdown', 'html'] = 'default', hint='<TailDisplay>'):
        super().__init__(hint)
        self.format = format
        self.deque = deque(maxlen=last_n)
    def log(self, line):
        self.deque.append(line)
        delim = '<br/>' if self.format == 'html' else '  \n'
        self.update(delim.join(self.deque), self.format)

class TableDisplay(TextDisplay):
    def __init__(self, last_n=1, hint='<TableDisplay>'):
        super().__init__(hint)
        self.deque = deque(maxlen=last_n)
    def log(self, record: namedtuple):
        record = record.__class__(*[f'{value:.4f}' if isinstance(value, float) else str(value) for value in record])
        header = '| ' + ' | '.join([field for field in record._fields]) + ' |\n' + '|-' * len(record._fields) + '|\n'
        row = '| ' + ' | '.join([f'{getattr(record, field)}' for field in record._fields]) + ' |' 
        self.deque.append(row)
        md = header + '  \n'.join(self.deque) + '\n'
        super().update(md, 'markdown')
    def update(self, records: Iterable[namedtuple]):
        self.deque.clear()
        for record in records:
            self.log(record)

class FigureDisplay(AbstractContextManager):
    def __init__(self, auto_close_fig=None, hint='<figure>'):
        self.figure_display = new_display(hint)
        self.fig = auto_close_fig
    def __exit__(self, exc_type, exc_value, traceback):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
        return False
    def show(self, fig=None):
        self.figure_display.update(fig if fig is not None else self.fig)