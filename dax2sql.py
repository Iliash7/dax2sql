#!/usr/bin/env python3
"""
dax2sql - translate a practical subset of DAX into SQL.

Usage
-----
  python dax2sql.py measures.dax
  python dax2sql.py -e "Total Sales = SUM(Sales[Amount])"
  python dax2sql.py query.dax -d tsql -r "Sales[ProductKey]=Product[ProductKey]"
  cat query.dax | python dax2sql.py -

Input may be
  * measure / calculated-column definitions:   Name = expr   |   Table[Name] = expr
  * query scripts:  [DEFINE MEASURE T[M] = ...]  EVALUATE <table expr> [ORDER BY ...]
  * a bare scalar expression

Relationships (-r) use the form  Many[col]=One[col  and drive JOIN generation.
Dialects: trino (default), ansi, postgres, mysql, sqlite, tsql.

DAX is a context-driven language and SQL is not, so this is a *translation of
common patterns*, not a full semantic equivalent.  Anything that can't be
translated faithfully (ALL() context removal, time intelligence, ...) is
reported as a WARNING comment in the output.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

DIALECTS = ("trino", "ansi", "postgres", "mysql", "sqlite", "tsql")


class DaxError(Exception):
    pass


class Unsupported(DaxError):
    pass


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|--[^\n]*|/\*.*?\*/)
  | (?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"]|"")*")
  | (?P<qident>'(?:[^']|'')*')
  | (?P<col>\[(?:[^\]]|\]\])*\])
  | (?P<op>&&|\|\||<>|<=|>=|==|[-+*/^&=<>(),{}])
  | (?P<ident>[^\W\d]\w*)
    """,
    re.X | re.S,
)


def tokenize(text):
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise DaxError(f"Unexpected character {text[pos]!r} at offset {pos}")
        pos = m.end()
        kind, val = m.lastgroup, m.group()
        if kind in ("ws", "comment"):
            continue
        if kind == "str":
            val = val[1:-1].replace('""', '"')
        elif kind == "qident":
            val = val[1:-1].replace("''", "'")
        elif kind == "col":
            val = val[1:-1].replace("]]", "]")
        out.append((kind, val))
    return out


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------
@dataclass
class Num:
    v: str


@dataclass
class Str:
    v: str


@dataclass
class Col:            # Table[Col]  or bare [Col] / [Measure]
    table: Optional[str]
    name: str


@dataclass
class Name:           # bare identifier: a table or a VAR
    name: str


@dataclass
class Func:
    name: str
    args: list


@dataclass
class Bin:
    op: str
    l: object
    r: object


@dataclass
class Un:
    op: str
    x: object


@dataclass
class SetLit:
    items: list


@dataclass
class Let:
    vars: list
    body: object


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
CMP = {"=", "==", "<>", "<", "<=", ">", ">="}


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else ("eof", "")

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def eof(self):
        return self.peek()[0] == "eof"

    def at_op(self, *ops):
        k, v = self.peek()
        return k == "op" and v in ops

    def at_kw(self, *kws):
        k, v = self.peek()
        return k == "ident" and v.upper() in kws

    def expect_op(self, op):
        if not self.at_op(op):
            raise DaxError(f"Expected {op!r} but found {self.peek()[1]!r}")
        self.next()

    def expect_kw(self, kw):
        if not self.at_kw(kw):
            raise DaxError(f"Expected {kw} but found {self.peek()[1]!r}")
        self.next()

    # ---- expressions (lowest -> highest precedence)
    def expr(self):
        if self.at_kw("VAR"):
            return self.let()
        return self.or_()

    def let(self):
        vars_ = []
        while self.at_kw("VAR"):
            self.next()
            name = self.next()[1]
            self.expect_op("=")
            vars_.append((name, self.or_()))
        self.expect_kw("RETURN")
        return Let(vars_, self.expr())

    def or_(self):
        l = self.and_()
        while self.at_op("||"):
            self.next()
            l = Bin("||", l, self.and_())
        return l

    def and_(self):
        l = self.cmp()
        while self.at_op("&&"):
            self.next()
            l = Bin("&&", l, self.cmp())
        return l

    def cmp(self):
        l = self.concat()
        while True:
            if self.at_op(*CMP):
                op = self.next()[1]
                l = Bin(op, l, self.concat())
            elif self.at_kw("IN"):
                self.next()
                l = Bin("IN", l, self.concat())
            else:
                return l

    def concat(self):
        l = self.add()
        while self.at_op("&"):
            self.next()
            l = Bin("&", l, self.add())
        return l

    def add(self):
        l = self.mul()
        while self.at_op("+", "-"):
            op = self.next()[1]
            l = Bin(op, l, self.mul())
        return l

    def mul(self):
        l = self.pow()
        while self.at_op("*", "/"):
            op = self.next()[1]
            l = Bin(op, l, self.pow())
        return l

    def pow(self):
        l = self.unary()
        while self.at_op("^"):
            self.next()
            l = Bin("^", l, self.unary())
        return l

    def unary(self):
        if self.at_op("-", "+"):
            op = self.next()[1]
            x = self.unary()
            return Un("-", x) if op == "-" else x
        return self.primary()

    def primary(self):
        k, v = self.next()
        if k == "num":
            return Num(v)
        if k == "str":
            return Str(v)
        if k == "col":
            return Col(None, v)
        if k == "op" and v == "(":
            e = self.expr()
            self.expect_op(")")
            return e
        if k == "op" and v == "{":
            items = []
            while not self.at_op("}"):
                items.append(self.expr())
                if self.at_op(","):
                    self.next()
            self.expect_op("}")
            return SetLit(items)
        if k in ("ident", "qident"):
            nk, nv = self.peek()
            if nk == "col":
                self.next()
                return Col(v, nv)
            if k == "ident" and nk == "op" and nv == "(":
                self.next()
                args = []
                if not self.at_op(")"):
                    while True:
                        args.append(self.expr())
                        if self.at_op(","):
                            self.next()
                            continue
                        break
                self.expect_op(")")
                return Func(v.upper(), args)
            if k == "ident" and v.upper() in ("TRUE", "FALSE"):
                return Func(v.upper(), [])
            return Name(v)
        raise DaxError(f"Unexpected token {v!r}")

    # ---- definitions
    def try_header(self):
        """Parse `Table[Name] =`, `[Name] =` or `Some Name =`; returns (table, name) or None."""
        save = self.i
        k, v = self.peek()
        if k == "col":
            self.next()
            table, name = None, v
        elif k in ("ident", "qident") and self.peek(1)[0] == "col":
            self.next()
            table, name = v, self.next()[1]
        elif k == "ident":
            parts = []
            while self.peek()[0] == "ident":
                parts.append(self.next()[1])
            table, name = None, " ".join(parts)
        else:
            return None
        if self.peek() == ("op", "="):
            self.next()
            return table, name
        self.i = save
        return None


# --------------------------------------------------------------------------
# SQL model
# --------------------------------------------------------------------------
@dataclass
class Select:
    items: Optional[list] = None          # None -> *
    tables: list = field(default_factory=list)
    sub: Optional["Select"] = None        # derived table source
    where: list = field(default_factory=list)
    group: list = field(default_factory=list)
    having: list = field(default_factory=list)
    order: list = field(default_factory=list)
    limit: Optional[int] = None
    distinct: bool = False
    aliases: set = field(default_factory=set)

    @property
    def derived(self):
        return self.sub is not None

    def add_table(self, t):
        if self.sub is None and t.lower() not in [x.lower() for x in self.tables]:
            self.tables.append(t)


def unparen(s):
    """Strip one pair of redundant outer parentheses."""
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth, in_str = 0, False
    for i, ch in enumerate(s):
        if ch == "'":
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s
    return s[1:-1]


RESERVED = {
    "date", "year", "month", "day", "order", "group", "user", "table", "key", "value",
    "index", "count", "select", "from", "where", "end", "case", "when", "name", "type",
    "time", "timestamp", "left", "right", "limit", "desc", "asc", "default", "check",
}

AGG = {"SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "COUNTROWS",
       "DISTINCTCOUNT", "DISTINCTCOUNTNOBLANK", "CONCATENATEX"}
ITER = {"SUMX", "AVERAGEX", "MINX", "MAXX", "COUNTX"}
TIME_INTEL = {"DATESYTD", "DATESMTD", "DATESQTD", "SAMEPERIODLASTYEAR", "DATEADD",
              "PARALLELPERIOD", "DATESBETWEEN", "DATESINPERIOD", "PREVIOUSMONTH",
              "PREVIOUSYEAR", "PREVIOUSQUARTER", "PREVIOUSDAY", "NEXTMONTH", "NEXTYEAR"}
CONTEXT_FILTERS = {"ALL", "ALLSELECTED", "ALLEXCEPT", "REMOVEFILTERS", "ALLNOBLANKROW"}
SQL_OPS = {"=": "=", "==": "=", "<>": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
           "&&": "AND", "||": "OR", "+": "+", "-": "-", "*": "*"}


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
class Gen:
    def __init__(self, dialect="ansi", relationships=()):
        self.d = dialect
        self.rels = list(relationships)     # (many_table, col, one_table, col)
        self.measures = {}                  # lower name -> (ast, table)
        self.gvars = {}                     # DEFINE VAR
        self.env = []                       # local VAR scopes
        self.filters = []                   # stack of CALCULATE filter layers
        self.used = {}                      # tables touched (ordered)
        self.default_table = None
        self.unq = False
        self.warnings = []
        self._inlining = []

    # ---- small helpers
    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def q(self, name):
        if re.fullmatch(r"[A-Za-z_]\w*", name) and name.lower() not in RESERVED:
            return name
        if self.d == "tsql":
            return "[" + name.replace("]", "]]") + "]"
        if self.d == "mysql":
            return "`" + name.replace("`", "``") + "`"
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def lit(v):
        return "'" + v.replace("'", "''") + "'"

    def use(self, table):
        self.used.setdefault(table.lower(), table)

    def tracked(self, fn, *args, propagate=True):
        saved, self.used = self.used, {}
        try:
            res = fn(*args)
            mine = list(self.used.values())
        finally:
            inner, self.used = self.used, saved
        if propagate:
            for k, v in inner.items():
                saved.setdefault(k, v)
        return res, mine

    def item(self, s, node):
        """Generate a scalar in the context of Select `s`, registering its tables."""
        old, self.unq = self.unq, s.derived
        try:
            res, tabs = self.tracked(self.sc, node)
        finally:
            self.unq = old
        for t in tabs:
            s.add_table(t)
        return res

    def cond_item(self, s, node):
        old, self.unq = self.unq, s.derived
        try:
            res, tabs = self.tracked(self.operand, node)
        finally:
            self.unq = old
        for t in tabs:
            s.add_table(t)
        return res

    def lookup_var(self, name):
        k = name.lower()
        for scope in reversed(self.env):
            if k in scope:
                return scope[k]
        return self.gvars.get(k)

    def concat(self, a, b):
        return f"CONCAT({a}, {b})" if self.d in ("mysql", "tsql") else f"({a} || {b})"

    def div(self, a, b):
        if self.d == "mysql":
            return f"({a} / {b})"
        if self.d == "trino":
            return f"(CAST({a} AS DOUBLE) / {b})"
        return f"(1.0 * {a} / {b})"

    def pairs(self, args):
        i = 0
        while i < len(args):
            if isinstance(args[i], Str) and i + 1 < len(args):
                yield args[i].v, args[i + 1]
                i += 2
            else:
                yield None, args[i]
                i += 1

    # ---- analysis
    def contains_agg(self, n, seen=()):
        if isinstance(n, Func):
            if n.name in ITER or (n.name in AGG and not (n.name in ("MIN", "MAX") and len(n.args) == 2)):
                return True
            return any(self.contains_agg(a, seen) for a in n.args)
        if isinstance(n, Col):
            m = self.measures.get(n.name.lower())
            if m and n.name.lower() not in seen:
                return self.contains_agg(m[0], seen + (n.name.lower(),))
            return False
        if isinstance(n, Name):
            v = self.lookup_var(n.name)
            return v is not None and self.contains_agg(v, seen)
        if isinstance(n, Bin):
            return self.contains_agg(n.l, seen) or self.contains_agg(n.r, seen)
        if isinstance(n, Un):
            return self.contains_agg(n.x, seen)
        if isinstance(n, SetLit):
            return any(self.contains_agg(a, seen) for a in n.items)
        if isinstance(n, Let):
            return any(self.contains_agg(v, seen) for _, v in n.vars) or self.contains_agg(n.body, seen)
        return False

    def refs_alias(self, n, aliases):
        if isinstance(n, Col):
            return n.table is None and n.name.lower() in aliases
        kids = []
        if isinstance(n, Func):
            kids = n.args
        elif isinstance(n, Bin):
            kids = [n.l, n.r]
        elif isinstance(n, Un):
            kids = [n.x]
        elif isinstance(n, SetLit):
            kids = n.items
        return any(self.refs_alias(k, aliases) for k in kids)

    def is_table_expr(self, n):
        if isinstance(n, Name):
            v = self.lookup_var(n.name)
            return self.is_table_expr(v) if v is not None else True
        if isinstance(n, Func):
            return hasattr(self, "t_" + n.name)
        if isinstance(n, Let):
            return self.is_table_expr(n.body)
        return False

    # ---- scalar generation
    def sc(self, n):
        return getattr(self, "s_" + type(n).__name__)(n)

    def operand(self, n):
        s = self.sc(n)
        if (isinstance(n, Bin) and (n.op in CMP or n.op == "IN")) or \
           (isinstance(n, Func) and n.name == "ISBLANK"):
            return unparen(s)
        return s

    def s_Num(self, n):
        return n.v

    def s_Str(self, n):
        return self.lit(n.v)

    def s_Col(self, n):
        m = self.measures.get(n.name.lower())
        if m and (n.table is None or m[1] is None or m[1].lower() == n.table.lower()):
            return self.inline_measure(n.name)
        table = n.table or self.default_table
        if table is None or self.unq:
            return self.q(n.name)
        self.use(table)
        return f"{self.q(table)}.{self.q(n.name)}"

    def inline_measure(self, name):
        key = name.lower()
        if key in self._inlining:
            raise DaxError(f"Circular measure reference: [{name}]")
        ast, table = self.measures[key]
        self._inlining.append(key)
        saved = (self.default_table, self.env)
        self.default_table, self.env = table, []
        try:
            return self.sc(ast)
        finally:
            self.default_table, self.env = saved
            self._inlining.pop()

    def s_Name(self, n):
        v = self.lookup_var(n.name)
        if v is None:
            raise Unsupported(f"Unknown identifier '{n.name}' (table used as a scalar?)")
        return self.sc(v)

    def s_Let(self, n):
        scope = {}
        self.env.append(scope)
        try:
            for name, node in n.vars:
                scope[name.lower()] = node
            return self.sc(n.body)
        finally:
            self.env.pop()

    def s_Un(self, n):
        return f"(-{self.sc(n.x)})"

    def s_SetLit(self, n):
        raise Unsupported("Set literal { } is only supported on the right of IN")

    def s_Bin(self, n):
        op = n.op
        if op in ("=", "==", "<>") and isinstance(n.r, Func) and n.r.name == "BLANK":
            return f"({self.sc(n.l)} IS {'NOT ' if op == '<>' else ''}NULL)"
        if op == "IN":
            l = self.sc(n.l)
            if isinstance(n.r, SetLit):
                rhs = "(" + ", ".join(self.sc(x) for x in n.r.items) + ")"
            else:
                rhs = "(\n" + textwrap.indent(self.render(self.to_select(n.r)), "    ") + "\n)"
            return f"({l} IN {rhs})"
        if op in ("&&", "||"):
            l, r = self.operand(n.l), self.operand(n.r)
        else:
            l, r = self.sc(n.l), self.sc(n.r)
        if op == "&":
            return self.concat(l, r)
        if op == "^":
            return f"POWER({l}, {r})"
        if op == "/":
            return self.div(l, r)
        return f"({l} {SQL_OPS[op]} {r})"

    def s_Func(self, n):
        m = getattr(self, "f_" + n.name, None)
        if m is None:
            if hasattr(self, "t_" + n.name):
                raise Unsupported(f"{n.name}() returns a table and can't be used as a scalar here")
            raise Unsupported(f"Function {n.name}() is not supported")
        return m(n.args)

    # ---- aggregation helpers
    def cond(self):
        flat = [c for layer in self.filters for c in layer]
        if not flat:
            return None
        return flat[0] if len(flat) == 1 else " AND ".join(flat)

    def agg(self, fn, inner, distinct=False):
        c = self.cond()
        if c:
            inner = f"CASE WHEN {c} THEN {inner} END"
        return f"{fn}({'DISTINCT ' if distinct else ''}{inner})"

    def with_conds(self, conds, fn, *args):
        self.filters.append(conds)
        try:
            return fn(*args)
        finally:
            self.filters.pop()

    def table_base(self, n):
        """Resolve a table expression to (table_name, [row conditions])."""
        if isinstance(n, Name):
            v = self.lookup_var(n.name)
            if v is not None:
                return self.table_base(v)
            self.use(n.name)
            return n.name, []
        if isinstance(n, Func):
            if n.name == "FILTER":
                t, c = self.table_base(n.args[0])
                return t, c + [self.operand(n.args[1])]
            if n.name in CONTEXT_FILTERS | {"RELATEDTABLE"} and n.args:
                a = n.args[0]
                t = a.table if isinstance(a, Col) else self.table_base(a)[0]
                self.use(t)
                return t, []
            if n.name == "CALCULATETABLE":
                t, c = self.table_base(n.args[0])
                for f in n.args[1:]:
                    c += self.filter_conds(f)
                return t, c
        raise Unsupported("Iterator/aggregate table argument must be a table, FILTER(...) or ALL(...)")

    def filter_conds(self, f):
        if isinstance(f, Func):
            nm = f.name
            if nm == "FILTER":
                t, c = self.table_base(f.args[0])
                return c + [self.operand(f.args[1])]
            if nm == "KEEPFILTERS":
                return self.filter_conds(f.args[0])
            if nm in CONTEXT_FILTERS:
                self.warn(f"{nm}() removes filter context; it has no direct SQL equivalent and was ignored "
                          "(consider a window function such as SUM(...) OVER ())")
                return []
            if nm in TIME_INTEL or nm in ("USERELATIONSHIP", "CROSSFILTER", "TREATAS"):
                self.warn(f"{nm}() is not translated and was ignored")
                return []
            if nm in ("VALUES", "DISTINCT"):
                self.warn(f"{nm}() used as a filter was ignored")
                return []
        return [self.operand(f)]

    # ---- aggregate functions
    def f_SUM(self, a):    return self.agg("SUM", self.sc(a[0]))
    def f_AVERAGE(self, a): return self.agg("AVG", self.sc(a[0]))
    def f_COUNT(self, a):  return self.agg("COUNT", self.sc(a[0]))
    def f_COUNTA(self, a): return self.agg("COUNT", self.sc(a[0]))
    def f_DISTINCTCOUNT(self, a): return self.agg("COUNT", self.sc(a[0]), distinct=True)
    def f_DISTINCTCOUNTNOBLANK(self, a): return self.f_DISTINCTCOUNT(a)

    def f_MIN(self, a):
        if len(a) == 2:
            return f"LEAST({self.sc(a[0])}, {self.sc(a[1])})"
        return self.agg("MIN", self.sc(a[0]))

    def f_MAX(self, a):
        if len(a) == 2:
            return f"GREATEST({self.sc(a[0])}, {self.sc(a[1])})"
        return self.agg("MAX", self.sc(a[0]))

    def f_COUNTROWS(self, a):
        _, conds = self.table_base(a[0])
        c = " AND ".join(conds) if conds else None
        return self.with_conds(conds, lambda: self._countrows())

    def _countrows(self):
        c = self.cond()
        return f"COUNT(CASE WHEN {c} THEN 1 END)" if c else "COUNT(*)"

    def _iter(self, fn, a):
        _, conds = self.table_base(a[0])
        return self.with_conds(conds, lambda: self.agg(fn, self.sc(a[1])))

    def f_SUMX(self, a):     return self._iter("SUM", a)
    def f_AVERAGEX(self, a): return self._iter("AVG", a)
    def f_MINX(self, a):     return self._iter("MIN", a)
    def f_MAXX(self, a):     return self._iter("MAX", a)
    def f_COUNTX(self, a):   return self._iter("COUNT", a)

    def f_CONCATENATEX(self, a):
        _, conds = self.table_base(a[0])

        def build():
            inner = self.sc(a[1])
            c = self.cond()
            if c:
                inner = f"CASE WHEN {c} THEN {inner} END"
            delim = self.sc(a[2]) if len(a) > 2 else "','"
            if self.d == "trino":
                return f"array_join(array_agg({inner}), {delim})"
            if self.d == "mysql":
                return f"GROUP_CONCAT({inner} SEPARATOR {delim})"
            if self.d == "sqlite":
                return f"GROUP_CONCAT({inner}, {delim})"
            if self.d == "ansi":
                return f"LISTAGG({inner}, {delim})"
            return f"STRING_AGG({inner}, {delim})"
        return self.with_conds(conds, build)

    # ---- context
    def f_CALCULATE(self, a):
        conds = []
        for f in a[1:]:
            conds += self.filter_conds(f)
        return self.with_conds(conds, lambda: self.sc(a[0]))

    def _timeintel(self, name, a):
        self.warn(f"{name}() time intelligence is not translated; the unfiltered expression was used")
        return self.sc(a[0])

    def f_TOTALYTD(self, a):  return self._timeintel("TOTALYTD", a)
    def f_TOTALMTD(self, a):  return self._timeintel("TOTALMTD", a)
    def f_TOTALQTD(self, a):  return self._timeintel("TOTALQTD", a)

    # ---- logic
    def f_IF(self, a):
        s = f"CASE WHEN {self.operand(a[0])} THEN {self.sc(a[1])}"
        if len(a) > 2:
            s += f" ELSE {self.sc(a[2])}"
        return s + " END"

    def f_SWITCH(self, a):
        subj, rest = a[0], a[1:]
        default = None
        if len(rest) % 2:
            default, rest = rest[-1], rest[:-1]
        true_mode = isinstance(subj, Func) and subj.name == "TRUE"
        s = "CASE" if true_mode else f"CASE {self.sc(subj)}"
        for v, r in zip(rest[0::2], rest[1::2]):
            s += f" WHEN {self.operand(v) if true_mode else self.sc(v)} THEN {self.sc(r)}"
        if default is not None:
            s += f" ELSE {self.sc(default)}"
        return s + " END"

    def f_AND(self, a): return f"({self.operand(a[0])} AND {self.operand(a[1])})"
    def f_OR(self, a):  return f"({self.operand(a[0])} OR {self.operand(a[1])})"
    def f_NOT(self, a): return f"(NOT ({self.operand(a[0])}))"
    def f_ISBLANK(self, a): return f"({self.sc(a[0])} IS NULL)"
    def f_BLANK(self, a): return "NULL"
    def f_COALESCE(self, a): return "COALESCE(" + ", ".join(self.sc(x) for x in a) + ")"

    def f_TRUE(self, a):  return "(1 = 1)" if self.d == "tsql" else "TRUE"
    def f_FALSE(self, a): return "(1 = 0)" if self.d == "tsql" else "FALSE"

    # ---- math
    def f_DIVIDE(self, a):
        core = self.div(self.sc(a[0]), f"NULLIF({self.sc(a[1])}, 0)")
        return f"COALESCE({core}, {self.sc(a[2])})" if len(a) > 2 else core

    def f_ABS(self, a):   return f"ABS({self.sc(a[0])})"
    def f_SQRT(self, a):  return f"SQRT({self.sc(a[0])})"
    def f_EXP(self, a):   return f"EXP({self.sc(a[0])})"
    def f_LN(self, a):    return f"LN({self.sc(a[0])})" if self.d != "tsql" else f"LOG({self.sc(a[0])})"
    def f_SIGN(self, a):  return f"SIGN({self.sc(a[0])})"
    def f_INT(self, a):   return f"FLOOR({self.sc(a[0])})"
    def f_POWER(self, a): return f"POWER({self.sc(a[0])}, {self.sc(a[1])})"

    def f_ROUND(self, a):
        return f"ROUND({self.sc(a[0])}, {self.sc(a[1]) if len(a) > 1 else '0'})"

    def f_MOD(self, a):
        x, y = self.sc(a[0]), self.sc(a[1])
        return f"({x} % {y})" if self.d == "tsql" else f"MOD({x}, {y})"

    # ---- text
    def f_UPPER(self, a): return f"UPPER({self.sc(a[0])})"
    def f_LOWER(self, a): return f"LOWER({self.sc(a[0])})"
    def f_TRIM(self, a):  return f"TRIM({self.sc(a[0])})"
    def f_LEN(self, a):   return f"{'LEN' if self.d == 'tsql' else 'LENGTH'}({self.sc(a[0])})"
    def f_LEFT(self, a):
        x, n = self.sc(a[0]), (self.sc(a[1]) if len(a) > 1 else "1")
        return f"SUBSTR({x}, 1, {n})" if self.d == "trino" else f"LEFT({x}, {n})"

    def f_RIGHT(self, a):
        x, n = self.sc(a[0]), (self.sc(a[1]) if len(a) > 1 else "1")
        return f"SUBSTR({x}, -{n})" if self.d == "trino" else f"RIGHT({x}, {n})"
    def f_MID(self, a):   return f"SUBSTRING({self.sc(a[0])}, {self.sc(a[1])}, {self.sc(a[2])})"
    def f_SUBSTITUTE(self, a): return f"REPLACE({self.sc(a[0])}, {self.sc(a[1])}, {self.sc(a[2])})"
    def f_CONCATENATE(self, a): return self.concat(self.sc(a[0]), self.sc(a[1]))

    def f_CONTAINSSTRING(self, a):
        pat = self.concat(self.concat("'%'", self.sc(a[1])), "'%'")
        return f"({self.sc(a[0])} LIKE {pat})"

    # ---- dates
    def _part(self, part, a):
        x = self.sc(a[0])
        if self.d == "tsql":
            return f"DATEPART({part}, {x})" if part == "QUARTER" else f"{part}({x})"
        if self.d == "mysql":
            return f"{part}({x})"
        if self.d == "sqlite":
            fmt = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}
            if part == "QUARTER":
                return f"((CAST(strftime('%m', {x}) AS INTEGER) + 2) / 3)"
            return f"CAST(strftime('{fmt[part]}', {x}) AS INTEGER)"
        return f"EXTRACT({part} FROM {x})"

    def f_YEAR(self, a):    return self._part("YEAR", a)
    def f_MONTH(self, a):   return self._part("MONTH", a)
    def f_DAY(self, a):     return self._part("DAY", a)
    def f_QUARTER(self, a): return self._part("QUARTER", a)

    def f_TODAY(self, a):
        return {"tsql": "CAST(GETDATE() AS DATE)", "sqlite": "DATE('now')"}.get(self.d, "CURRENT_DATE")

    def f_NOW(self, a):
        return {"tsql": "GETDATE()", "sqlite": "DATETIME('now')"}.get(self.d, "CURRENT_TIMESTAMP")

    def f_DATE(self, a):
        y, m, d = (self.sc(x) for x in a)
        if self.d == "trino":
            return f"CAST(format('%04d-%02d-%02d', {y}, {m}, {d}) AS DATE)"
        if self.d == "tsql":
            return f"DATEFROMPARTS({y}, {m}, {d})"
        if self.d == "mysql":
            return f"STR_TO_DATE(CONCAT({y}, '-', {m}, '-', {d}), '%Y-%m-%d')"
        if self.d == "sqlite":
            return f"DATE(printf('%04d-%02d-%02d', {y}, {m}, {d}))"
        return f"MAKE_DATE({y}, {m}, {d})"

    def f_DATEDIFF(self, a):
        start, end = self.sc(a[0]), self.sc(a[1])
        unit = (a[2].name if isinstance(a[2], Name) else self.sc(a[2])).upper()
        if self.d == "trino":
            return f"date_diff('{unit.lower()}', {start}, {end})"
        if self.d == "tsql":
            return f"DATEDIFF({unit}, {start}, {end})"
        if self.d == "mysql":
            return f"TIMESTAMPDIFF({unit}, {start}, {end})"
        if unit == "DAY":
            if self.d == "sqlite":
                return f"CAST(julianday({end}) - julianday({start}) AS INTEGER)"
            return f"(CAST({end} AS DATE) - CAST({start} AS DATE))"
        raise Unsupported(f"DATEDIFF unit {unit} is not supported for dialect {self.d}")

    # ---- relationships / lookups
    def f_RELATED(self, a):
        return self.sc(a[0])

    def f_LOOKUPVALUE(self, a):
        s = Select(items=[])
        s.items.append(self.item(s, a[0]))
        for col, val in zip(a[1::2], a[2::2]):
            c = self.item(s, col)
            s.where.append(f"{c} = {self.sc(val)}")
        return "(" + self.render(s).replace("\n", " ") + ")"

    # ---- table generation
    def to_select(self, n):
        if isinstance(n, Name):
            v = self.lookup_var(n.name)
            return self.to_select(v) if v is not None else Select(tables=[n.name])
        if isinstance(n, Let):
            self.env.append({})
            try:
                for name, node in n.vars:
                    self.env[-1][name.lower()] = node
                return self.to_select(n.body)
            finally:
                self.env.pop()
        if isinstance(n, Func) and hasattr(self, "t_" + n.name):
            return getattr(self, "t_" + n.name)(n.args)
        raise Unsupported("Expression is not a supported table expression")

    def wrap(self, s):
        w = Select(sub=s)
        w.aliases = set(s.aliases)
        return w

    def t_ALL(self, a):
        arg = a[0]
        if isinstance(arg, Col):
            ref = self.item(s := Select(tables=[arg.table] if arg.table else []), arg)
            s.items, s.group = [ref], [ref]
            s.aliases.add(arg.name.lower())
            return s
        return self.to_select(arg)

    t_ALLSELECTED = t_ALLNOBLANKROW = t_ALL

    def t_VALUES(self, a):
        arg = a[0]
        if isinstance(arg, Col):
            return self.t_ALL(a)
        s = self.to_select(arg)
        s.distinct = True
        return s

    t_DISTINCT = t_VALUES

    def t_FILTER(self, a):
        s = self.to_select(a[0])
        cond = a[1]
        if self.contains_agg(cond) and s.group:
            s.having.append(self.cond_item(s, cond))
            return s
        if s.limit is not None or self.refs_alias(cond, s.aliases):
            s = self.wrap(s)
        s.where.append(self.cond_item(s, cond))
        return s

    def t_CALCULATETABLE(self, a):
        s = self.to_select(a[0])
        if s.limit is not None:
            s = self.wrap(s)
        for f in a[1:]:
            old, self.unq = self.unq, s.derived
            try:
                conds, tabs = self.tracked(self.filter_conds, f)
            finally:
                self.unq = old
            for t in tabs:
                s.add_table(t)
            s.where += conds
        return s

    def t_SUMMARIZE(self, a):
        s = self.to_select(a[0])
        if s.group or s.limit is not None or s.items is not None or s.distinct:
            s = self.wrap(s)
        s.items = []
        for name, node in self.pairs(a[1:]):
            if name is None:
                if isinstance(node, Col):
                    ref = self.item(s, node)
                    s.items.append(ref)
                    s.group.append(ref)
                    s.aliases.add(node.name.lower())
                else:
                    self.warn(f"Unsupported SUMMARIZE argument ({type(node).__name__}) ignored")
            else:
                s.items.append(f"{self.item(s, node)} AS {self.q(name)}")
                s.aliases.add(name.lower())
        return s

    def t_SUMMARIZECOLUMNS(self, a):
        s = Select(items=[])
        for name, node in self.pairs(a):
            if name is not None:
                s.items.append(f"{self.item(s, node)} AS {self.q(name)}")
                s.aliases.add(name.lower())
            elif isinstance(node, Col):
                ref = self.item(s, node)
                s.items.append(ref)
                s.group.append(ref)
                s.aliases.add(node.name.lower())
            elif isinstance(node, Func) and node.name in ("IGNORE", "NONVISUAL", "ROLLUPADDISSUBTOTAL", "ROLLUP"):
                self.warn(f"{node.name}() was ignored")
            else:
                conds, tabs = self.tracked(self.filter_conds, node)
                for t in tabs:
                    s.add_table(t)
                s.where += conds
        return s

    def t_ADDCOLUMNS(self, a):
        s = self.to_select(a[0])
        if s.items is None:
            s.items = ["*"]
        for name, node in self.pairs(a[1:]):
            if name is None:
                continue
            if self.contains_agg(node) and not s.group:
                self.warn("Aggregate inside ADDCOLUMNS over a plain table: DAX evaluates it per row; "
                          "the SQL result is a single grand total unless you GROUP BY the key")
            s.items.append(f"{self.item(s, node)} AS {self.q(name)}")
            s.aliases.add(name.lower())
        return s

    def t_SELECTCOLUMNS(self, a):
        s = self.to_select(a[0])
        if s.group or s.limit is not None:
            s = self.wrap(s)
        s.items = []
        s.aliases = set()
        for name, node in self.pairs(a[1:]):
            expr = self.item(s, node)
            if name is None and isinstance(node, Col):
                s.items.append(expr)
                s.aliases.add(node.name.lower())
            elif name is not None:
                s.items.append(f"{expr} AS {self.q(name)}")
                s.aliases.add(name.lower())
        return s

    def t_ROW(self, a):
        s = Select(items=[])
        for name, node in self.pairs(a):
            if name is not None:
                s.items.append(f"{self.item(s, node)} AS {self.q(name)}")
                s.aliases.add(name.lower())
        return s

    def t_TOPN(self, a):
        n = int(float(self.sc(a[0])))
        s = self.to_select(a[1])
        if s.limit is not None:
            s = self.wrap(s)
        rest, i = a[2:], 0
        while i < len(rest):
            expr, desc = rest[i], True
            i += 1
            if i < len(rest):
                d = rest[i]
                if isinstance(d, Name) and d.name.upper() in ("ASC", "DESC"):
                    desc, i = d.name.upper() == "DESC", i + 1
                elif isinstance(d, Num):
                    desc, i = d.v == "0", i + 1
                elif isinstance(d, Func) and d.name in ("TRUE", "FALSE"):
                    desc, i = d.name == "FALSE", i + 1
            if self.contains_agg(expr) and not s.group:
                self.warn("TOPN ordering by an aggregate over an ungrouped table; add a GROUP BY on the key")
            s.order.append(self.order_item(s, expr, desc))
        s.limit = n
        return s

    def order_item(self, s, node, desc):
        if isinstance(node, Col) and node.table is None and node.name.lower() in s.aliases:
            txt = self.q(node.name)
        else:
            txt = self.item(s, node)
        return f"{txt} DESC" if desc else txt

    # ---- rendering
    def find_rel(self, t, joined):
        lt = t.lower()
        jl = {j.lower(): j for j in joined}
        for (mt, mc, ot, oc) in self.rels:
            if mt.lower() == lt and ot.lower() in jl:
                return mc, jl[ot.lower()], oc
            if ot.lower() == lt and mt.lower() in jl:
                return oc, jl[mt.lower()], mc
        return None

    def build_from(self, tables):
        tabs = list(tables)
        if len(tabs) == 1:
            return [f"FROM {self.q(tabs[0])}"], tabs[0]
        low = {t.lower() for t in tabs}
        one_side = {r[2].lower() for r in self.rels if r[0].lower() in low and r[2].lower() in low}
        base = next((t for t in tabs if t.lower() not in one_side), tabs[0])
        joined, rest = [base], [t for t in tabs if t is not base]
        lines = [f"FROM {self.q(base)}"]
        while rest:
            progressed = False
            for t in list(rest):
                rel = self.find_rel(t, joined)
                if rel:
                    tc, other, oc = rel
                    lines.append(f"INNER JOIN {self.q(t)} ON {self.q(t)}.{self.q(tc)} = "
                                 f"{self.q(other)}.{self.q(oc)}")
                    joined.append(t)
                    rest.remove(t)
                    progressed = True
            if not progressed:
                t = rest.pop(0)
                self.warn(f"No relationship known between {t} and {', '.join(joined)}; "
                          "emitted CROSS JOIN (pass -r Many[col]=One[col])")
                lines.append(f"CROSS JOIN {self.q(t)}")
                joined.append(t)
        return lines, base

    def render(self, s):
        base, from_lines = None, []
        if s.sub is not None:
            from_lines = ["FROM (", textwrap.indent(self.render(s.sub), "    "), ") AS t"]
        elif s.tables:
            from_lines, base = self.build_from(s.tables)
        items = ["*"] if s.items is None else s.items
        star = f"{self.q(base)}.*" if base and len(s.tables) > 1 else "*"
        items = [star if i == "*" else i for i in items]
        head = "SELECT"
        if s.distinct:
            head += " DISTINCT"
        if s.limit is not None and self.d == "tsql":
            head += f" TOP {s.limit}"
        if len(items) <= 1:
            lines = [f"{head} {items[0] if items else '1'}"]
        else:
            lines = [head + "\n" + ",\n".join("    " + i for i in items)]
        lines += from_lines
        if s.where:
            lines.append("WHERE " + "\n  AND ".join(s.where))
        if s.group:
            lines.append("GROUP BY " + ", ".join(s.group))
        if s.having:
            lines.append("HAVING " + "\n   AND ".join(s.having))
        if s.order:
            lines.append("ORDER BY " + ", ".join(s.order))
        if s.limit is not None and self.d != "tsql":
            lines.append(f"FETCH FIRST {s.limit} ROWS ONLY" if self.d == "ansi" else f"LIMIT {s.limit}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Top-level conversion
# --------------------------------------------------------------------------
_REL = re.compile(r"^\s*'?([^\[']+?)'?\s*\[(.+?)\]\s*=\s*'?([^\[']+?)'?\s*\[(.+?)\]\s*$")


def parse_relationship(text):
    m = _REL.match(text)
    if not m:
        raise DaxError(f"Bad relationship {text!r}; expected Many[col]=One[col]")
    return m.groups()


def convert(text, dialect="trino", relationships=()):
    """Return (sql_text, ok)."""
    g = Gen(dialect, relationships)
    p = Parser(tokenize(text))
    defs, evals = [], []

    if p.at_kw("DEFINE"):
        p.next()
        while p.at_kw("MEASURE", "COLUMN", "VAR", "TABLE"):
            kw = p.next()[1].upper()
            if kw == "VAR":
                name = p.next()[1]
                p.expect_op("=")
                g.gvars[name.lower()] = p.expr()
                continue
            hdr = p.try_header()
            if not hdr:
                raise DaxError(f"Expected a name after DEFINE {kw}")
            defs.append((hdr[1], hdr[0], p.expr()))
    while p.at_kw("EVALUATE"):
        p.next()
        ast, order = p.expr(), []
        if p.at_kw("ORDER"):
            p.next()
            p.expect_kw("BY")
            while True:
                node = p.expr()
                desc = False
                if p.at_kw("ASC", "DESC"):
                    desc = p.next()[1].upper() == "DESC"
                order.append((node, desc))
                if p.at_op(","):
                    p.next()
                    continue
                break
        if p.at_kw("START"):
            g.warn("START AT is not supported and was ignored")
            while not p.eof() and not p.at_kw("EVALUATE"):
                p.next()
        evals.append((ast, order))

    if not defs and not evals:
        while not p.eof():
            hdr = p.try_header()
            if hdr:
                defs.append((hdr[1], hdr[0], p.expr()))
            else:
                evals.append((p.expr(), []))
                break
    if not p.eof():
        raise DaxError(f"Unexpected trailing input near {p.peek()[1]!r}")

    for name, table, ast in defs:
        g.measures[name.lower()] = (ast, table)

    jobs = []
    if evals:
        jobs = [("Query", lambda a=a, o=o: eval_sql(g, a, o)) for a, o in evals]
    else:
        jobs = [(name, lambda d=d: def_sql(g, *d)) for d in defs for name in [d[0]]]

    out, ok = [], True
    for label, fn in jobs:
        g.warnings, g.default_table, g.env, g.filters, g.used = [], None, [], [], {}
        try:
            sql = fn()
            body = "\n".join(f"-- WARNING: {w}" for w in g.warnings)
            out.append((f"{body}\n" if body else "") + f"{sql};")
        except DaxError as e:
            ok = False
            out.append(f"-- ERROR converting {label}: {e}")
    header_sep = "\n\n"
    return header_sep.join(out), ok


def eval_sql(g, ast, order):
    if g.is_table_expr(ast):
        s = g.to_select(ast)
    else:
        s = Select(items=[g.item(Select(), ast)])
        s.tables = list(g.used.values())
    for node, desc in order:
        s.order.append(g.order_item(s, node, desc))
    return g.render(s)


def def_sql(g, name, table, ast):
    g.default_table = table
    label = f"-- {'Column' if table and not g.contains_agg(ast) else 'Measure'}: " + \
            (f"{table}[{name}]" if table else name)
    if g.is_table_expr(ast):
        s = g.to_select(ast)
    elif table and not g.contains_agg(ast):          # calculated column
        s = Select(tables=[table], items=["*"])
        s.items.append(f"{g.item(s, ast)} AS {g.q(name)}")
    else:                                            # measure
        s = Select(items=[])
        s.items.append(f"{g.item(s, ast)} AS {g.q(name)}")
    return label + "\n" + g.render(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Translate DAX to SQL.")
    ap.add_argument("input", nargs="?", help="DAX file, or '-' for stdin")
    ap.add_argument("-e", "--expr", help="DAX text given on the command line")
    ap.add_argument("-d", "--dialect", choices=DIALECTS, default="trino")
    ap.add_argument("-r", "--rel", action="append", default=[],
                    help="relationship Many[col]=One[col] (repeatable)")
    ap.add_argument("-o", "--output", help="write SQL to this file")
    args = ap.parse_args(argv)

    if args.expr:
        text = args.expr
    elif args.input in (None, "-"):
        text = sys.stdin.read()
    else:
        with open(args.input, encoding="utf-8") as fh:
            text = fh.read()
    try:
        rels = [parse_relationship(r) for r in args.rel]
        sql, ok = convert(text, args.dialect, rels)
    except DaxError as e:
        print(f"-- ERROR: {e}", file=sys.stderr)
        return 1
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(sql + "\n")
    else:
        print(sql)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
