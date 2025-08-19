# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.
#
# Modifications:
# - Converted all list values to numpy arrays (dtype=object).
# - Replaced Bio.File.as_handle with a simplified internal handle manager.

"""Turn an mmCIF file into a dictionary."""

import numpy as np


class _SimpleHandle:
    """Simplified replacement for Bio.File.as_handle."""

    def __init__(self, file):
        self.file = file
        self.handle = None
        self.should_close = False

    def __enter__(self):
        if hasattr(self.file, "read"):
            self.handle = self.file  # already an open handle
            self.should_close = False
        else:
            self.handle = open(self.file, "rt", encoding="utf-8", errors="replace")
            self.should_close = True
        return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.should_close and self.handle:
            self.handle.close()


class MMCIF2Dict(dict):
    """Parse a mmCIF file and return a dictionary."""

    def __init__(self, filename):
        """Parse a mmCIF file and return a dictionary.

        Arguments
        ---------
        filename : str or file handle
            Name of the mmCIF file OR an open file handle.
        """
        self.quote_chars = ["'", '"']
        self.whitespace_chars = [" ", "\t"]

        with _SimpleHandle(filename) as handle:
            loop_flag = False
            key = None
            tokens = self._tokenize(handle)
            try:
                token = next(tokens)
            except StopIteration:
                return
            self[token[0:5]] = token[5:]
            if not token[0:5].startswith("data_"):
                raise ValueError(
                    "The input mmCIF file must begin with a 'data_' directive."
                )
            i = 0
            n = 0
            for token in tokens:
                if token.lower() == "loop_":
                    loop_flag = True
                    keys = []
                    i = 0
                    n = 0
                    continue
                elif loop_flag:
                    if token.startswith("_") and (n == 0 or i % n == 0):
                        if i > 0:
                            loop_flag = False
                        else:
                            self[token] = []
                            keys.append(token)
                            n += 1
                            continue
                    else:
                        self[keys[i % n]].append(token)
                        i += 1
                        continue
                if key is None:
                    key = token
                else:
                    self[key] = [token]
                    key = None

        # Convert all list values to numpy arrays
        for k, v in list(self.items()):
            if isinstance(v, list):
                self[k] = np.asarray(v, dtype=object)

    def _splitline(self, line):
        in_token = False
        quote_open_char = None
        start_i = 0
        for i, c in enumerate(line):
            if c in self.whitespace_chars:
                if in_token and not quote_open_char:
                    in_token = False
                    yield line[start_i:i]
            elif c in self.quote_chars:
                if not quote_open_char and not in_token:
                    quote_open_char = c
                    in_token = True
                    start_i = i + 1
                elif c == quote_open_char and (
                    i + 1 == len(line) or line[i + 1] in self.whitespace_chars
                ):
                    quote_open_char = None
                    in_token = False
                    yield line[start_i:i]
            elif c == "#" and not in_token:
                return
            elif not in_token:
                in_token = True
                start_i = i
        if in_token:
            yield line[start_i:]
        if quote_open_char:
            raise ValueError("Line ended with quote open: " + line)

    def _tokenize(self, handle):
        empty = True
        for line in handle:
            empty = False
            if line.startswith("#"):
                continue
            elif line.startswith(";"):
                token_buffer = [line[1:].rstrip()]
                for line in handle:
                    line = line.rstrip()
                    if line.startswith(";"):
                        yield "\n".join(token_buffer)
                        line = line[1:]
                        if line and line[0] not in self.whitespace_chars:
                            raise ValueError("Missing whitespace")
                        break
                    token_buffer.append(line)
                else:
                    raise ValueError("Missing closing semicolon")
            yield from self._splitline(line.strip())
        if empty:
            raise ValueError("Empty file.")
