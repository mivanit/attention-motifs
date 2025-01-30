"""pydbg is an implementation of the Rust builtin `dbg!` for Python.

from https://github.com/tylerwince/pydbg/blob/master/pydbg.py
"""

import os
import inspect
import sys
import typing
from pathlib import Path


__version__ = "0.3.0"


_ExpType = typing.TypeVar("_ExpType")

cwd: Path = Path.cwd().absolute()


def dbg(exp: _ExpType) -> _ExpType:
	"""Call dbg with any variable or expression.

	Calling dbg will print to stderr the current filename and lineno,
	as well as the passed expression and what the expression evaluates to:

		from pydbg import dbg

		a = 2
		b = 5

		dbg(a+b)

		def square(x: int) -> int:
			return x * x

		dbg(square(a))

	"""

	for frame in inspect.stack():
		line = frame.code_context[0]
		if "dbg" in line:
			start = line.find("(") + 1
			end = line.rfind(")")
			if end == -1:
				end = len(line)

			file: Path = Path(frame.filename).absolute()
			common = Path(os.path.commonpath([file, cwd]))
			fname: str = file.relative_to(common).as_posix()

			print(
				f"[{fname}:{frame.lineno}] {line[start:end]} = {exp!r}",
				file=sys.stderr,
			)
			break

	return exp
