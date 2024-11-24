from collections import deque, namedtuple
from IPython.display import display
from . import NopContextManager

class TailDisplay(NopContextManager):
    def __init__(self, last_n=1):
        self.tail_display = display(display_id=True)
        self.deque = deque(maxlen=last_n)
    def log(self, line):
        self.deque.append(line)
        md = '  \n'.join(self.deque)
        from IPython.display import Markdown
        self.tail_display.update(Markdown(md))

class TableDisplay(NopContextManager):
    def __init__(self, last_n=1):
        self.tail_display = display('<markdown display>', display_id=True)
        self.deque = deque(maxlen=last_n)
    def log(self, record: namedtuple):
        record = record.__class__(*[f'{value:.4f}' if isinstance(value, float) else str(value) for value in record])
        header = '| ' + ' | '.join([field for field in record._fields]) + ' |\n' + '|-' * len(record._fields) + '|\n'
        row = '| ' + ' | '.join([f'{getattr(record, field)}' for field in record._fields]) + ' |' 
        self.deque.append(row)
        md = header + '  \n'.join(self.deque) + '\n'
        from IPython.display import Markdown
        self.tail_display.update(Markdown(md))
