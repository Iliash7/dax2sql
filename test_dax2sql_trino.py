"""
Test-suite for dax2sql, Trino dialect.

How the tests work
------------------
Expected values come from DAX semantics and a small hand-computed dataset (below),
NOT from the program's own output.

Three kinds of checks:
  1. SYNTAX    every generated statement is parsed with sqlglot's *Trino* parser.
  2. SEMANTICS the Trino SQL is transpiled to DuckDB and executed on the dataset;
               results are compared with the value DAX would return.
  3. TRINO     string-level checks for Trino-specific constructs (LIMIT, date_diff, ...).

Caveat: no Trino server is available here, so (2) runs on DuckDB. Where DuckDB is
more lenient than Trino (e.g. `'a' || 1`), a string-level test covers the gap.

Run:  python -m pytest test_dax2sql_trino.py -v
"""
import os
import re
import sys
import unittest
from decimal import Decimal

import duckdb
import sqlglot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dax2sql import DaxError, convert, parse_relationship  # noqa: E402

REL = [parse_relationship("Sales[ProductKey]=Product[ProductKey]")]

# ---------------------------------------------------------------------------
# Dataset (all expected numbers in the tests are derived from this by hand)
#
#  SaleID Prod Region Year Amount Cost Customer OrderDate   ShipDate(+5d)
#   1      1   West   2023  100     60  Ann      2023-01-10
#   2      1   East   2023  200    120  Bob      2023-02-11
#   3      2   West   2024  300    150  Ann      2024-03-12
#   4      2   West   2024  400    200  Cy       2024-04-13
#   5      3   East   2024   50     40  Bob      2024-05-14
#   6      3   North  2022  500    250  Dee      2022-06-15
#   7      1   NULL   2024   10      5  Eve      2024-07-16
#
#  Totals: Amount 1560, Cost 825.   Region: West 800, East 250, North 500, NULL 10
#  Year:   2022 500, 2023 300, 2024 760.   Category: Tools 1010, Toys 550
#  Customer: Ann 400, Bob 250, Cy 400, Dee 500, Eve 10
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE Sales (SaleID INT, ProductKey INT, Region VARCHAR, "Year" INT, Amount INT,
                    Cost INT, Customer VARCHAR, OrderDate DATE, ShipDate DATE);
INSERT INTO Sales VALUES
 (1,1,'West',2023,100, 60,'Ann','2023-01-10','2023-01-15'),
 (2,1,'East',2023,200,120,'Bob','2023-02-11','2023-02-16'),
 (3,2,'West',2024,300,150,'Ann','2024-03-12','2024-03-17'),
 (4,2,'West',2024,400,200,'Cy' ,'2024-04-13','2024-04-18'),
 (5,3,'East',2024, 50, 40,'Bob','2024-05-14','2024-05-19'),
 (6,3,'North',2022,500,250,'Dee','2022-06-15','2022-06-20'),
 (7,1,NULL ,2024, 10,  5,'Eve','2024-07-16','2024-07-21');
CREATE TABLE Product (ProductKey INT, Name VARCHAR, Category VARCHAR);
INSERT INTO Product VALUES (1,'Widget','Tools'),(2,'Gadget','Tools'),(3,'Doohickey','Toys');
"""


def split_statements(sql):
    return [p for p in re.split(r";\s*(?:\n\s*\n|$)", sql.strip()) if p.strip()]


def norm(sql):
    """Strip comment lines and collapse whitespace."""
    body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    return re.sub(r"\s+", " ", body).strip()


def same(a, b):
    if isinstance(a, Decimal):
        a = float(a)
    if isinstance(b, Decimal):
        b = float(b)
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-9
    return a == b


class Result:
    def __init__(self, trino, duck, cols, rows):
        self.trino, self.duck, self.cols, self.rows = trino, duck, cols, rows

    def dicts(self):
        return [dict(zip(self.cols, r)) for r in self.rows]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect()
        cls.con.execute(SCHEMA)

    # -- helpers
    def execute(self, dax, rels=REL):
        sql, ok = convert(dax, "trino", rels)
        self.assertTrue(ok, f"convert() reported an error:\n{sql}")
        out = []
        for st in split_statements(sql):
            body = "\n".join(l for l in st.splitlines() if not l.strip().startswith("--"))
            try:
                tree = sqlglot.parse_one(body, read="trino")
            except Exception as e:  # noqa: BLE001
                self.fail(f"Output is not valid Trino SQL: {e}\n{body}")
            duck = tree.sql(dialect="duckdb")
            try:
                cur = self.con.execute(duck)
            except Exception as e:  # noqa: BLE001
                self.fail(f"Execution failed: {e}\n--- Trino SQL ---\n{body}\n--- DuckDB ---\n{duck}")
            out.append(Result(body, duck, [d[0] for d in cur.description], cur.fetchall()))
        return out

    def assertValue(self, dax, expected, idx=0, rels=REL):
        res = self.execute(dax, rels)[idx]
        self.assertEqual(len(res.rows), 1, f"expected 1 row, got {res.rows}\n{res.trino}")
        got = res.rows[0][0]
        if expected is None:
            self.assertIsNone(got, res.trino)
        else:
            self.assertTrue(same(got, expected), f"expected {expected!r}, got {got!r}\n{res.trino}")

    def assertRows(self, dax, expected, ordered=True, rels=REL, idx=0):
        res = self.execute(dax, rels)[idx]
        got = [tuple(r) for r in res.rows]
        if not ordered:
            key = lambda r: tuple(str(x) for x in r)  # noqa: E731
            got, expected = sorted(got, key=key), sorted(expected, key=key)
        self.assertEqual(len(got), len(expected), f"row count: got {got}\n{res.trino}")
        for g, e in zip(got, expected):
            self.assertEqual(len(g), len(e), f"column count: got {g}, want {e}\n{res.trino}")
            for gv, ev in zip(g, e):
                self.assertTrue(same(gv, ev), f"row mismatch: got {g}, want {e}\n{res.trino}")


# ===========================================================================
# 1. Aggregates
# ===========================================================================
class TestAggregates(Base):
    def test_sum(self):              self.assertValue("x = SUM(Sales[Amount])", 1560)
    def test_average(self):          self.assertValue("x = AVERAGE(Sales[Amount])", 1560 / 7)
    def test_min(self):              self.assertValue("x = MIN(Sales[Amount])", 10)
    def test_max(self):              self.assertValue("x = MAX(Sales[Amount])", 500)
    def test_count_ignores_blank(self):  self.assertValue("x = COUNT(Sales[Region])", 6)
    def test_counta(self):           self.assertValue("x = COUNTA(Sales[Region])", 6)
    def test_countrows(self):        self.assertValue("x = COUNTROWS(Sales)", 7)
    def test_distinctcount(self):    self.assertValue("x = DISTINCTCOUNT(Sales[Customer])", 5)
    def test_function_names_case_insensitive(self):
        self.assertValue("x = sum(Sales[Amount])", 1560)
        self.assertValue("x = Sum(Sales[Amount])", 1560)

    def test_sumx_expression(self):
        self.assertValue("x = SUMX(Sales, Sales[Amount] - Sales[Cost])", 735)

    def test_sumx_over_filter(self):
        self.assertValue("x = SUMX(FILTER(Sales, Sales[Year] = 2024), Sales[Amount])", 760)

    def test_averagex_over_filter(self):
        self.assertValue("x = AVERAGEX(FILTER(Sales, Sales[Year] = 2024), Sales[Amount])", 190)

    def test_maxx_over_filter(self):
        self.assertValue('x = MAXX(FILTER(Sales, Sales[Region] = "East"), Sales[Amount])', 200)

    def test_countrows_over_filter(self):
        self.assertValue("x = COUNTROWS(FILTER(Sales, Sales[Amount] > 100))", 4)

    def test_concatenatex_single(self):
        self.assertValue('x = CONCATENATEX(FILTER(Sales, Sales[Year] = 2022), Sales[Customer], ", ")', "Dee")

    def test_concatenatex_multi(self):
        res = self.execute('x = CONCATENATEX(FILTER(Sales, Sales[Year] = 2023), Sales[Customer], ", ")')[0]
        self.assertEqual(sorted(res.rows[0][0].split(", ")), ["Ann", "Bob"])


# ===========================================================================
# 2. CALCULATE / filter context
# ===========================================================================
class TestCalculate(Base):
    def test_single_filter(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "West")', 800)

    def test_two_filters_are_anded(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "West", Sales[Year] = 2024)', 700)

    def test_in_set(self):
        self.assertValue("x = CALCULATE(SUM(Sales[Amount]), Sales[Year] IN {2022, 2023})", 800)

    def test_in_set_strings(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Sales[Region] IN {"East", "North"})', 750)

    def test_filter_function_with_or(self):
        self.assertValue(
            'x = CALCULATE(SUM(Sales[Amount]), FILTER(Sales, Sales[Region] = "West" || Sales[Region] = "East"))', 1050)

    def test_comparison_operators(self):
        self.assertValue("x = CALCULATE(SUM(Sales[Amount]), Sales[Amount] >= 300)", 1200)
        self.assertValue("x = CALCULATE(SUM(Sales[Amount]), Sales[Amount] < 100)", 60)
        self.assertValue("x = CALCULATE(SUM(Sales[Amount]), Sales[Year] <> 2024)", 800)

    def test_nested_calculate_accumulates_filters(self):
        self.assertValue(
            'x = CALCULATE(CALCULATE(SUM(Sales[Amount]), Sales[Region] = "West"), Sales[Year] = 2024)', 700)

    def test_empty_result_is_blank(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "South")', None)

    def test_blank_equality(self):
        self.assertValue("x = CALCULATE(COUNTROWS(Sales), Sales[Region] = BLANK())", 1)

    def test_blank_inequality(self):
        self.assertValue("x = CALCULATE(COUNTROWS(Sales), Sales[Region] <> BLANK())", 6)

    def test_filter_on_related_table(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Product[Category] = "Toys")', 550)

    def test_filters_across_two_tables(self):
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Product[Category] = "Tools", Sales[Year] = 2023)', 300)

    def test_measure_references(self):
        script = """
        Total = SUM(Sales[Amount])
        West  = CALCULATE([Total], Sales[Region] = "West")
        Share = DIVIDE([West], [Total])
        """
        self.assertValue(script, 1560, idx=0)
        self.assertValue(script, 800, idx=1)
        self.assertValue(script, 800 / 1560, idx=2)

    def test_measure_inside_arithmetic(self):
        script = """
        Rev = SUM(Sales[Amount])
        Cst = SUM(Sales[Cost])
        Profit = [Rev] - [Cst]
        """
        self.assertValue(script, 735, idx=2)

    def test_qualified_measure_reference(self):
        script = """
        Sales[Total] = SUM(Sales[Amount])
        Sales[Twice] = Sales[Total] * 2
        """
        self.assertValue(script, 3120, idx=1)


# ===========================================================================
# 3. Scalar expressions (no tables needed)
# ===========================================================================
class TestScalars(Base):
    def test_precedence(self):          self.assertValue("x = 1 + 2 * 3 - 4", 3)
    def test_parentheses(self):         self.assertValue("x = (1 + 2) * (3 - 4)", -3)
    def test_power_and_division(self):  self.assertValue("x = 2 ^ 3 + 10 / 4", 10.5)
    def test_unary_minus(self):         self.assertValue("x = -(2 + 3)", -5)
    def test_division_is_not_integer(self):  self.assertValue("x = 1 / 4", 0.25)
    def test_divide_function(self):     self.assertValue("x = DIVIDE(1, 4)", 0.25)
    def test_divide_by_zero_blank(self): self.assertValue("x = DIVIDE(1, 0)", None)
    def test_divide_by_zero_alt(self):  self.assertValue("x = DIVIDE(1, 0, -1)", -1)

    def test_if_true(self):             self.assertValue('x = IF(1 < 2, "yes", "no")', "yes")
    def test_if_false(self):            self.assertValue('x = IF(1 > 2, "yes", "no")', "no")
    def test_if_without_else(self):     self.assertValue('x = IF(1 > 2, "yes")', None)

    def test_and_or_not_functions(self):
        self.assertValue('x = IF(AND(1 = 1, OR(1 = 2, NOT(2 = 3))), "ok", "no")', "ok")

    def test_logical_operators(self):
        self.assertValue('x = IF(1 <> 2 && (3 < 2 || 4 >= 4), "y", "n")', "y")

    def test_logical_precedence_and_over_or(self):
        # TRUE || (FALSE && FALSE) -> TRUE ;  (TRUE || FALSE) && FALSE would be FALSE
        self.assertValue('x = IF(1 = 1 || 1 = 2 && 1 = 2, "y", "n")', "y")

    def test_switch_value_form(self):
        self.assertValue('x = SWITCH(2, 1, "one", 2, "two", "other")', "two")

    def test_switch_default(self):
        self.assertValue('x = SWITCH(9, 1, "one", 2, "two", "other")', "other")

    def test_switch_true_form(self):
        self.assertValue('x = SWITCH(TRUE(), 5 > 10, "A", 5 > 1, "B", "C")', "B")

    def test_switch_over_aggregate(self):
        self.assertValue('x = SWITCH(TRUE(), SUM(Sales[Amount]) > 5000, "A", SUM(Sales[Amount]) > 1000, "B", "C")', "B")

    def test_coalesce_and_isblank(self):
        self.assertValue("x = COALESCE(BLANK(), 5)", 5)
        self.assertValue('x = IF(ISBLANK(BLANK()), "b", "n")', "b")

    def test_math_functions(self):
        self.assertValue("x = MOD(10, 3)", 1)
        self.assertValue("x = INT(3.7)", 3)
        self.assertValue("x = INT(-3.7)", -4)         # DAX INT rounds down
        self.assertValue("x = ROUND(2.567, 1)", 2.6)
        self.assertValue("x = ABS(-4)", 4)
        self.assertValue("x = SQRT(16)", 4)
        self.assertValue("x = POWER(2, 10)", 1024)

    def test_two_argument_max_min(self):
        self.assertValue("x = MAX(3, 7)", 7)
        self.assertValue("x = MIN(3, 7)", 3)

    def test_string_escaping(self):
        self.assertValue('x = "it\'s"', "it's")
        self.assertValue('x = "say ""hi"""', 'say "hi"')

    def test_string_concat(self):
        self.assertValue('x = "a" & "b"', "ab")

    def test_comments_ignored(self):
        self.assertValue("x = 1 + /* c */ 2 // trailing\n", 3)
        self.assertValue("-- leading comment\nx = 1 + 2", 3)

    def test_var_return(self):
        self.assertValue("x = VAR t = SUM(Sales[Amount]) VAR u = SUM(Sales[Cost]) RETURN t - u", 735)

    def test_var_used_twice(self):
        self.assertValue('x = VAR t = SUM(Sales[Amount]) RETURN IF(t > 1000, t, 0)', 1560)

    def test_bare_expression_input(self):
        self.assertValue("SUM(Sales[Amount]) + 1", 1561)


# ===========================================================================
# 4. Calculated columns
# ===========================================================================
class TestColumns(Base):
    def col(self, dax, name):
        res = self.execute(dax)[0]
        self.assertEqual(len(res.rows), 7, res.trino)
        return {d["SaleID"]: d[name] for d in res.dicts()}

    def test_arithmetic(self):
        c = self.col("Sales[Profit] = Sales[Amount] - Sales[Cost]", "Profit")
        self.assertEqual(c[1], 40)
        self.assertEqual(sum(c.values()), 735)

    def test_keeps_original_columns(self):
        res = self.execute("Sales[Profit] = Sales[Amount] - Sales[Cost]")[0]
        for name in ("SaleID", "Region", "Amount", "Cost", "Customer"):
            self.assertIn(name.lower(), [c.lower() for c in res.cols])

    def test_year_month(self):
        c = self.col("Sales[YM] = YEAR(Sales[OrderDate]) * 100 + MONTH(Sales[OrderDate])", "YM")
        self.assertEqual(c[1], 202301)
        self.assertEqual(c[5], 202405)

    def test_day_and_quarter(self):
        self.assertEqual(self.col("Sales[D] = DAY(Sales[OrderDate])", "D")[3], 12)
        self.assertEqual(self.col("Sales[Q] = QUARTER(Sales[OrderDate])", "Q")[5], 2)
        self.assertEqual(self.col("Sales[Q] = QUARTER(Sales[OrderDate])", "Q")[7], 3)

    def test_datediff(self):
        c = self.col("Sales[Lag] = DATEDIFF(Sales[OrderDate], Sales[ShipDate], DAY)", "Lag")
        self.assertEqual(set(c.values()), {5})

    def test_text_functions(self):
        self.assertEqual(self.col("Sales[I] = UPPER(LEFT(Sales[Customer], 2))", "I")[1], "AN")
        self.assertEqual(self.col("Sales[L] = LEN(Sales[Customer])", "L")[4], 2)
        self.assertEqual(self.col("Sales[L] = LOWER(Sales[Customer])", "L")[2], "bob")

    def test_right(self):
        self.assertEqual(self.col("Sales[R] = RIGHT(Sales[Customer], 1)", "R")[1], "n")

    def test_substitute(self):
        self.assertEqual(self.col('Sales[S] = SUBSTITUTE(Sales[Customer], "n", "N")', "S")[1], "ANN")

    def test_mid(self):
        self.assertEqual(self.col("Sales[S] = MID(Sales[Customer], 2, 2)", "S")[1], "nn")

    def test_margin_uses_float_division(self):
        c = self.col("Sales[M] = DIVIDE(Sales[Amount] - Sales[Cost], Sales[Amount])", "M")
        self.assertAlmostEqual(float(c[1]), 0.4)
        self.assertAlmostEqual(float(c[6]), 0.5)

    def test_coalesce_column(self):
        c = self.col('Sales[R2] = COALESCE(Sales[Region], "none")', "R2")
        self.assertEqual(c[7], "none")
        self.assertEqual(c[1], "West")

    def test_if_isblank_column(self):
        c = self.col("Sales[B] = IF(ISBLANK(Sales[Region]), 1, 0)", "B")
        self.assertEqual(c[7], 1)
        self.assertEqual(c[1], 0)

    def test_switch_column(self):
        c = self.col('Sales[Size] = SWITCH(TRUE(), Sales[Amount] >= 400, "L", Sales[Amount] >= 100, "M", "S")', "Size")
        self.assertEqual([c[i] for i in (1, 4, 5, 6, 7)], ["M", "L", "S", "L", "S"])

    def test_related_column_needs_join(self):
        c = self.col("Sales[Cat] = RELATED(Product[Category])", "Cat")
        self.assertEqual(c[1], "Tools")
        self.assertEqual(c[5], "Toys")

    def test_lookupvalue(self):
        c = self.col("Sales[PN] = LOOKUPVALUE(Product[Name], Product[ProductKey], Sales[ProductKey])", "PN")
        self.assertEqual(c[1], "Widget")
        self.assertEqual(c[6], "Doohickey")


# ===========================================================================
# 5. EVALUATE table queries
# ===========================================================================
class TestQueries(Base):
    def test_summarizecolumns_join_order_by(self):
        self.assertRows('EVALUATE SUMMARIZECOLUMNS(Product[Category], "S", SUM(Sales[Amount])) ORDER BY [S] DESC',
                        [("Tools", 1010), ("Toys", 550)])

    def test_summarizecolumns_order_asc(self):
        self.assertRows('EVALUATE SUMMARIZECOLUMNS(Product[Category], "S", SUM(Sales[Amount])) ORDER BY [S]',
                        [("Toys", 550), ("Tools", 1010)])

    def test_summarize(self):
        self.assertRows('EVALUATE SUMMARIZE(Sales, Sales[Customer], "T", SUM(Sales[Amount])) ORDER BY Sales[Customer]',
                        [("Ann", 400), ("Bob", 250), ("Cy", 400), ("Dee", 500), ("Eve", 10)])

    def test_order_by_desc_column(self):
        self.assertRows('EVALUATE SUMMARIZE(Sales, Sales[Year], "T", SUM(Sales[Amount])) ORDER BY Sales[Year] DESC',
                        [(2024, 760), (2023, 300), (2022, 500)])

    def test_topn_desc(self):
        self.assertRows('EVALUATE TOPN(2, SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])), [T], DESC)',
                        [("West", 800), ("North", 500)])

    def test_topn_asc(self):
        self.assertRows('EVALUATE TOPN(2, SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])), [T], ASC)',
                        [(None, 10), ("East", 250)], ordered=False)

    def test_topn_default_is_descending(self):
        self.assertRows('EVALUATE TOPN(1, SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])), [T])',
                        [("West", 800)])

    def test_topn_from_plain_table_ordered_by_column(self):
        self.assertRows("EVALUATE SELECTCOLUMNS(TOPN(2, Sales, Sales[Amount], DESC), \"A\", Sales[Amount])",
                        [(500,), (400,)], ordered=False)

    def test_filter_table(self):
        res = self.execute('EVALUATE FILTER(Sales, Sales[Amount] > 100 && (Sales[Region] = "West" || Sales[Region] = "East"))')[0]
        self.assertEqual({d["SaleID"] for d in res.dicts()}, {2, 3, 4})

    def test_filter_with_aggregate_becomes_having(self):
        res = self.execute("EVALUATE FILTER(VALUES(Sales[Customer]), SUM(Sales[Amount]) > 300)")[0]
        self.assertEqual({r[0] for r in res.rows}, {"Ann", "Cy", "Dee"})

    def test_values_includes_blank_row(self):
        res = self.execute("EVALUATE VALUES(Sales[Region])")[0]
        self.assertEqual({r[0] for r in res.rows}, {"West", "East", "North", None})
        self.assertEqual(len(res.rows), 4)

    def test_distinct(self):
        res = self.execute("EVALUATE DISTINCT(Sales[Year])")[0]
        self.assertEqual(sorted(r[0] for r in res.rows), [2022, 2023, 2024])

    def test_all_column(self):
        res = self.execute("EVALUATE ALL(Sales[Customer])")[0]
        self.assertEqual(len(res.rows), 5)

    def test_addcolumns_with_measure(self):
        self.assertRows("""DEFINE MEASURE Sales[Total] = SUM(Sales[Amount])
                           EVALUATE ADDCOLUMNS(VALUES(Sales[Year]), "T", [Total])""",
                        [(2022, 500), (2023, 300), (2024, 760)], ordered=False)

    def test_selectcolumns(self):
        self.assertRows('EVALUATE SELECTCOLUMNS(FILTER(Sales, Sales[Year] = 2022), "Cust", Sales[Customer], "Amt", Sales[Amount])',
                        [("Dee", 500)])

    def test_row(self):
        self.assertRows('EVALUATE ROW("Total", SUM(Sales[Amount]), "Cnt", COUNTROWS(Sales))', [(1560, 7)])

    def test_calculatetable(self):
        res = self.execute("EVALUATE CALCULATETABLE(Sales, Sales[Year] = 2022)")[0]
        self.assertEqual([d["SaleID"] for d in res.dicts()], [6])

    def test_summarizecolumns_with_filter_argument(self):
        self.assertRows('''EVALUATE SUMMARIZECOLUMNS(Sales[Region], FILTER(ALL(Sales), Sales[Year] = 2024),
                           "T", SUM(Sales[Amount]))''',
                        [("West", 700), ("East", 50), (None, 10)], ordered=False)

    def test_summarizecolumns_two_group_columns_across_tables(self):
        self.assertRows('''EVALUATE SUMMARIZECOLUMNS(Product[Category], Sales[Year], "T", SUM(Sales[Amount]))
                           ORDER BY Sales[Year], Product[Category]''',
                        [("Toys", 2022, 500), ("Tools", 2023, 300),
                         ("Tools", 2024, 710), ("Toys", 2024, 50)])

    def test_multiple_evaluate_blocks(self):
        res = self.execute("EVALUATE ROW(\"a\", 1)\nEVALUATE ROW(\"b\", 2)")
        self.assertEqual([r.rows[0][0] for r in res], [1, 2])

    def test_in_with_table_expression(self):
        self.assertValue("x = CALCULATE(SUM(Sales[Amount]), Sales[Customer] IN VALUES(Sales[Customer]))", 1560)


# ===========================================================================
# 6. Trino-specific output (string level)
# ===========================================================================
class TestTrinoSyntax(unittest.TestCase):
    def t(self, dax, rels=REL):
        sql, ok = convert(dax, "trino", rels)
        self.assertTrue(ok, sql)
        return norm(sql)

    def test_simple_measure_exact(self):
        self.assertEqual(self.t("Total Sales = SUM(Sales[Amount])"),
                         'SELECT SUM(Sales.Amount) AS "Total Sales" FROM Sales;')

    def test_calculate_exact(self):
        self.assertEqual(self.t('West Sales = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "West")'),
                         "SELECT SUM(CASE WHEN Sales.Region = 'West' THEN Sales.Amount END) "
                         'AS "West Sales" FROM Sales;')

    def test_topn_uses_limit(self):
        s = self.t('EVALUATE TOPN(3, SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])), [T], DESC)')
        self.assertTrue(s.rstrip(";").endswith("LIMIT 3"), s)
        self.assertNotRegex(s, r"\bTOP\b")
        self.assertNotIn("FETCH FIRST", s)

    def test_order_by_desc(self):
        s = self.t('EVALUATE SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])) ORDER BY [T] DESC')
        self.assertIn("ORDER BY T DESC", s)

    def test_double_quote_identifiers_only(self):
        s = self.t("Total Sales = MAX(Sales[Year])")
        self.assertIn('"Year"', s)
        self.assertIn('"Total Sales"', s)
        self.assertNotIn("`", s)
        self.assertNotRegex(s, r"\[")

    def test_divide_casts_to_double_and_guards_zero(self):
        s = self.t("x = DIVIDE(Sales[Amount], Sales[Cost])")
        self.assertIn("AS DOUBLE", s)
        self.assertIn("NULLIF(Sales.Cost, 0)", s)

    def test_slash_operator_casts_to_double(self):
        self.assertIn("AS DOUBLE", self.t("x = 1 / 4"))

    def test_no_left_right_functions(self):
        s = self.t("Sales[X] = LEFT(Sales[Customer], 2) & RIGHT(Sales[Customer], 1)")
        self.assertNotRegex(s, r"\bLEFT\(")
        self.assertNotRegex(s, r"\bRIGHT\(")

    def test_concat_operator(self):
        self.assertIn("||", self.t('x = "a" & "b"'))
        self.assertNotIn("CONCAT(", self.t('x = "a" & "b"'))

    def test_number_in_concat_is_cast_to_varchar(self):
        # Trino's || only accepts varchar operands; DAX & converts numbers implicitly.
        self.assertRegex(self.t('x = "n=" & 5'), r"CAST\(\s*5\s+AS\s+VARCHAR")

    def test_date_function(self):
        self.assertIn("CAST(format('%04d-%02d-%02d', 2024, 1, 2) AS DATE)", self.t("x = DATE(2024, 1, 2)"))

    def test_datediff(self):
        self.assertIn("date_diff('day', Sales.OrderDate, Sales.ShipDate)",
                      self.t("x = DATEDIFF(Sales[OrderDate], Sales[ShipDate], DAY)"))

    def test_extract_for_date_parts(self):
        s = self.t("x = YEAR(Sales[OrderDate]) + MONTH(Sales[OrderDate])")
        self.assertIn("EXTRACT(YEAR FROM Sales.OrderDate)", s)
        self.assertIn("EXTRACT(MONTH FROM Sales.OrderDate)", s)

    def test_today_now(self):
        self.assertIn("CURRENT_DATE", self.t("x = TODAY()"))
        self.assertIn("CURRENT_TIMESTAMP", self.t("x = NOW()"))

    def test_concatenatex_uses_array_join(self):
        s = self.t('x = CONCATENATEX(Sales, Sales[Customer], ", ")')
        self.assertIn("array_join(array_agg(Sales.Customer), ', ')", s)
        self.assertNotIn("STRING_AGG", s)
        self.assertNotIn("GROUP_CONCAT", s)

    def test_join_generated_from_relationship(self):
        s = self.t('EVALUATE SUMMARIZECOLUMNS(Product[Category], "S", SUM(Sales[Amount]))')
        self.assertIn("FROM Sales INNER JOIN Product ON Product.ProductKey = Sales.ProductKey", s)
        self.assertIn("GROUP BY Product.Category", s)

    def test_fact_table_is_base_even_if_dimension_mentioned_first(self):
        s = self.t('EVALUATE SUMMARIZECOLUMNS(Product[Category], "S", SUM(Sales[Amount]))')
        self.assertTrue(re.search(r"FROM Sales INNER JOIN", s), s)

    def test_string_literals_use_single_quotes(self):
        self.assertIn("'it''s'", self.t('x = "it\'s"'))

    def test_boolean_literals_are_native(self):
        s = self.t("x = IF(TRUE(), 1, 0)")
        self.assertIn("TRUE", s)
        self.assertNotIn("1 = 1", s)

    def test_null_literal(self):
        self.assertIn("IS NULL", self.t("x = IF(ISBLANK(Sales[Region]), 1, 0)"))

    def test_default_dialect_is_trino(self):
        sql_default, _ = convert('EVALUATE TOPN(1, SUMMARIZE(Sales, Sales[Region], "T", SUM(Sales[Amount])), [T])')
        self.assertIn("LIMIT 1", sql_default)


# ===========================================================================
# 7. Warnings and errors
# ===========================================================================
class TestDiagnostics(unittest.TestCase):
    def test_unsupported_function_reports_error(self):
        sql, ok = convert("x = FOO(1)", "trino")
        self.assertFalse(ok)
        self.assertIn("-- ERROR", sql)
        self.assertIn("FOO", sql)

    def test_one_bad_statement_does_not_hide_the_good_one(self):
        sql, ok = convert("a = SUM(Sales[Amount])\nb = FOO(1)", "trino")
        self.assertFalse(ok)
        self.assertIn("SUM(Sales.Amount)", sql)
        self.assertIn("-- ERROR", sql)

    def test_all_warns_instead_of_silently_dropping(self):
        sql, ok = convert("x = CALCULATE(SUM(Sales[Amount]), ALL(Sales))", "trino")
        self.assertTrue(ok)
        self.assertIn("WARNING", sql)
        self.assertIn("ALL", sql)

    def test_removefilters_warns(self):
        sql, _ = convert("x = CALCULATE(SUM(Sales[Amount]), REMOVEFILTERS(Sales[Region]))", "trino")
        self.assertIn("WARNING", sql)

    def test_time_intelligence_warns(self):
        sql, _ = convert("x = TOTALYTD(SUM(Sales[Amount]), Sales[OrderDate])", "trino")
        self.assertIn("WARNING", sql)
        sql, _ = convert("x = CALCULATE(SUM(Sales[Amount]), SAMEPERIODLASTYEAR(Sales[OrderDate]))", "trino")
        self.assertIn("WARNING", sql)

    def test_missing_relationship_warns_and_cross_joins(self):
        sql, ok = convert('EVALUATE SUMMARIZECOLUMNS(Product[Category], "S", SUM(Sales[Amount]))', "trino", [])
        self.assertIn("CROSS JOIN", sql)
        self.assertIn("WARNING", sql)

    def test_circular_measures_detected(self):
        sql, ok = convert("A = [B] + 1\nB = [A] + 1", "trino")
        self.assertFalse(ok)
        self.assertIn("ircular", sql)

    def test_filter_as_definition_body_is_a_table_query(self):
        sql, ok = convert("x = FILTER(Sales, Sales[Amount] > 1)", "trino")
        # FILTER as a definition body is a table expression, so it must convert, not crash
        self.assertTrue(ok, sql)

    def test_syntax_error_raises(self):
        with self.assertRaises(DaxError):
            convert("x = SUM(", "trino")

    def test_unknown_character_raises(self):
        with self.assertRaises(DaxError):
            convert("x = 5 $ 3", "trino")

    def test_unbalanced_paren_raises(self):
        with self.assertRaises(DaxError):
            convert("x = (1 + 2", "trino")


class TestRelationshipParsing(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_relationship("Sales[ProductKey]=Product[ProductKey]"),
                         ("Sales", "ProductKey", "Product", "ProductKey"))

    def test_quoted_table_and_spaces(self):
        self.assertEqual(parse_relationship("'Sales Data'[Key] = Dim[Key]"),
                         ("Sales Data", "Key", "Dim", "Key"))

    def test_invalid(self):
        with self.assertRaises(DaxError):
            parse_relationship("Sales.ProductKey = Product.ProductKey")


# ===========================================================================
# 8. DAX semantics that plain SQL does NOT share.
#    These assert what DAX actually returns.  They are expected to expose gaps.
# ===========================================================================
class TestDaxSemanticsVsSql(Base):
    def test_string_comparison_is_case_insensitive_in_dax(self):
        # DAX: "West" = "west" is TRUE
        self.assertValue('x = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "west")', 800)

    def test_containsstring_is_case_insensitive_in_dax(self):
        res = self.execute('Sales[Has] = CONTAINSSTRING(Sales[Customer], "AN")')[0]
        row1 = [d for d in res.dicts() if d["SaleID"] == 1][0]
        self.assertTrue(row1["Has"])       # "Ann" contains "an" (case-insensitive)

    def test_ampersand_treats_blank_as_empty_string(self):
        res = self.execute('Sales[T] = Sales[Customer] & "-" & Sales[Region]')[0]
        row7 = [d for d in res.dicts() if d["SaleID"] == 7][0]
        self.assertEqual(row7["T"], "Eve-")   # DAX: BLANK & "x" = "x"

    def test_distinctcount_counts_blank_as_a_value(self):
        # DAX DISTINCTCOUNT counts BLANK: West, East, North, (blank) = 4
        self.assertValue("x = DISTINCTCOUNT(Sales[Region])", 4)

    def test_summarizecolumns_drops_rows_where_all_measures_are_blank(self):
        # DAX omits the 2022 row because [West] is blank there.
        self.assertRows('''DEFINE MEASURE Sales[West] = CALCULATE(SUM(Sales[Amount]), Sales[Region] = "West")
                           EVALUATE SUMMARIZECOLUMNS(Sales[Year], "W", [West]) ORDER BY Sales[Year]''',
                        [(2023, 100), (2024, 700)])

    def test_distinctcountnoblank_skips_blank(self):
        self.assertValue("x = DISTINCTCOUNTNOBLANK(Sales[Region])", 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
