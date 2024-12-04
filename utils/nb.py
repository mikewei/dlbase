from typing import Literal, Iterable, Optional, NamedTuple
from collections import deque, namedtuple
from itertools import chain
from contextlib import AbstractContextManager
from IPython.display import display, DisplayHandle
from .py import NopContextManager

__all__ = ['new_display', 'TextDisplay', 'TailDisplay', 'TableDisplay', 'FigureDisplay']

def new_display(hint=None) -> Optional[DisplayHandle]:
    return display(display_id=True) if hint is None else display(hint, display_id=True)

class TextDisplay(NopContextManager):
    def __init__(self, hint='<TextDisplay>'):
        self.display = new_display(hint)
        if self.display is None:
            raise RuntimeError('new_display() returned None')
    def update(self, text, format: Literal['default', 'markdown', 'html'] = 'default'):
        if self.display is None:
            return
        if format == 'default':
            self.display.update(text)
        elif format == 'markdown':
            from IPython.core.display import Markdown
            self.display.update(Markdown(text))
        elif format == 'html':
            from IPython.core.display import HTML
            self.display.update(HTML(text))

class TailDisplay(TextDisplay):
    def __init__(self, last_n=1, format: Literal['default', 'markdown', 'html'] = 'default', hint='<TailDisplay>'):
        super().__init__(hint)
        self.format: Literal['default', 'markdown', 'html'] = format
        self.deque = deque(maxlen=last_n)
    def log(self, line):
        self.deque.append(line)
        delim = '<br/>' if self.format == 'html' else '  \n'
        self.update(delim.join(self.deque), self.format)

class TableDisplay(TextDisplay):
    def __init__(self, last_n=1, hint='<TableDisplay>'):
        super().__init__(hint)
        self.deque = deque(maxlen=last_n)
        self.RecType = None
    def log(self, *records: NamedTuple):
        if self.RecType is None:
            self.RecType = namedtuple('RecType', chain(*(r._fields for r in records)))
        rec = self.RecType(*[f'{value:.4f}' if isinstance(value, float) else str(value) for record in records for value in record])
        header = '| ' + ' | '.join([field for field in rec._fields]) + ' |\n' + '|-' * len(rec._fields) + '|\n'
        row = '| ' + ' | '.join([f'{getattr(rec, field)}' for field in rec._fields]) + ' |' 
        self.deque.append(row)
        md = header + '  \n'.join(self.deque) + '\n'
        super().update(md, 'markdown')
    def update(self, records: Iterable[NamedTuple]):
        self.deque.clear()
        for record in records:
            self.log(record)

class FigureDisplay(AbstractContextManager):
    def __init__(self, auto_close_fig=None, hint='<figure>'):
        self.fig = auto_close_fig
        self.figure_display = new_display(hint)
        if self.figure_display is None:
            self.cleanup()
            raise RuntimeError('new_display() returned None')
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False
    def cleanup(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
    def show(self, fig=None):
        if self.figure_display is None:
            return
        self.figure_display.update(fig if fig is not None else self.fig)