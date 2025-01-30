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

_CWD: Path = Path.cwd().absolute()

_COUNTER: int = 0

_NoExpPassed = object()


def dbg(exp: _ExpType = _NoExpPassed) -> _ExpType:
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
	global _COUNTER

	for frame in inspect.stack():
		line = frame.code_context[0]
		if "dbg" in line:
			start = line.find("(") + 1
			end = line.rfind(")")
			if end == -1:
				end = len(line)

			file: Path = Path(frame.filename).absolute()
			common = Path(os.path.commonpath([file, _CWD]))
			fname: str = file.relative_to(common).as_posix()

			msg: str = f"[{fname}:{frame.lineno}]"

			if exp is _NoExpPassed:
				msg += f" (dbg {_COUNTER})"
			else:
				msg += f" {line[start:end]} = {exp!r}"
				_COUNTER += 1
			print(
				msg,
				file=sys.stderr,
			)
			break

	return exp
