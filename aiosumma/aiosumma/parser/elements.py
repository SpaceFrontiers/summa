# -*- coding: utf-8 -*-
"""Elements that will constitute the parse tree of a query.
You may use these items to build a tree representing a query,
or get a tree as the result of parsing a query string.
"""
import re
from decimal import Decimal

from aiosumma.errors import UnsupportedQueryError

_MARKER = object()
NON_DEFAULT_QUOTES = r''''“”‘«»„`'''
QUOTES = NON_DEFAULT_QUOTES + '"'
QUOTE_RE = re.compile(f'[{QUOTES}]')
NON_DEFAULT_QUOTE_RE = re.compile(f'[{NON_DEFAULT_QUOTES}]')


class Item:
    """Base class for all items that compose the parse tree.
    An item is a part of a request.
    """

    _equality_attrs = []

    @property
    def children(self):
        """As base of a tree structure, an item may have children"""
        return []

    def __repr__(self):
        children = ", ".join(c.__repr__() for c in self.children)
        return "%s(%s)" % (self.__class__.__name__, children)

    def __eq__(self, other):
        """a basic equal operation
        """
        return (self.__class__ == other.__class__
                and len(self.children) == len(other.children)
                and all(
                    getattr(self, a, _MARKER) == getattr(other, a, _MARKER)
                    for a in self._equality_attrs
                )
                and all(c.__eq__(d) for c, d in zip(self.children, other.children)))

    def to_summa_query(self):
        return {'match': {'value': str(self)}}


class SearchField(Item):
    """Indicate wich field the search expression operates on
    eg: *desc* in ``desc:(this OR that)``
    :param str name: name of the field
    :param expr: the searched expression
    """
    _equality_attrs = ['name']

    def __init__(self, name, expr):
        super().__init__()
        self.name = name
        self.expr = expr

    def __str__(self):
        return self.name + ":" + self.expr.__str__()

    def __repr__(self):
        return "SearchField(%r, %s)" % (self.name, self.expr.__repr__())

    def to_summa_query(self):
        if isinstance(self.expr, Range):
            return {'range': {'field': self.name, 'value': self.expr.to_partial_summa_query()}}
        elif isinstance(self.expr, Word):
            return {'term': {'field': self.name, 'value': self.expr.value}}
        elif isinstance(self.expr, Phrase):
            return {'phrase': {'field': self.name, 'value': self.expr.value}}
        elif isinstance(self.expr, Regex):
            return {'regex': {'field': self.name, 'value': self.expr.value}}
        elif isinstance(self.expr, Proximity):
            return {'phrase': {'field': self.name, 'value': self.expr.term.value, 'slop': self.expr.slop}}
        else:
            raise UnsupportedQueryError(error=f'{self.expr} in search field `{self.name}`')

    @property
    def children(self):
        """the only child is the expression"""
        return [self.expr]


class BaseGroup(Item):
    """
    Parent class for binary operations are binary operation used to join expressions,
    like OR and AND
    :param operands: expressions to apply operation on
    """
    def __init__(self, *operands):
        super().__init__()
        self.operands = operands

    def __str__(self):
        return f'({" ".join(str(o) for o in self.operands)})'

    def __len__(self):
        return len(self.operands)

    def to_summa_query(self):
        subqueries = []
        if not self.operands:
            return {'all': {}}
        for operand in self.operands:
            if isinstance(operand, Plus):
                subqueries.append({'occur': 'must', 'query': operand.a.to_summa_query()})
            elif isinstance(operand, Minus):
                subqueries.append({'occur': 'must_not', 'query': operand.a.to_summa_query()})
            else:
                query = operand.to_summa_query()
                if query:
                    subqueries.append({'occur': 'should', 'query': query})
        return {'boolean': {'subqueries': subqueries}}

    @property
    def children(self):
        """children are left and right expressions
        """
        return self.operands


class Group(BaseGroup):
    pass


class SynonymsGroup(BaseGroup):
    def to_summa_query(self):
        disjuncts = []
        if not self.operands:
            return {'all': {}}
        for operand in self.operands:
            query = operand.to_summa_query()
            if query:
                disjuncts.append(query)
        return {'disjunction_max': {'disjuncts': disjuncts}}


class Range(Item):
    """A Range
    :param left: left bound
    :param right: right bound
    :param bool including_left: wether left bound is included
    :param bool including_right: wether right bound is included
    """

    LEFT_CHAR = {True: '[', False: '{'}
    RIGHT_CHAR = {True: ']', False: '}'}

    def __init__(self, left, right, including_left=True, including_right=True):
        super().__init__()
        self.left = left
        self.right = right
        self.including_left = including_left
        self.including_right = including_right

    @property
    def children(self):
        """children are lower and higher bound expressions"""
        return [self.left, self.right]

    def __str__(self):
        return "%s%s TO %s%s" % (
            self.LEFT_CHAR[self.including_left],
            self.left.__str__(),
            self.right.__str__(),
            self.RIGHT_CHAR[self.including_right])

    def to_partial_summa_query(self):
        return {
            'left': str(self.left),
            'right': str(self.right),
            'including_left': self.including_left,
            'including_right': self.including_right
        }


class Term(Item):
    """Base for terms
    :param str value: the value
    """
    WILDCARDS_PATTERN = re.compile(r"((?<=[^\\])[?*]|^[?*])")  # non escaped * and ?
    # see
    # https://lucene.apache.org/core/3_6_0/queryparsersyntax.html#Escaping%20Special%20Characters
    WORD_ESCAPED_CHARS = re.compile(r'\\([+\-&|!(){}[\]^"~*?:\\])')

    _equality_attrs = ['value']

    def __init__(self, value):
        super().__init__()
        self.value = value

    @property
    def unescaped_value(self):
        # remove '\' that escape characters
        return self.WORD_ESCAPED_CHARS.sub(r'\1', self.value)

    def is_wildcard(self):
        """:return bool: True if value is the wildcard ``*``
        """
        return self.value == "*"

    def iter_wildcards(self):
        """list wildcards contained in value and their positions
        """
        for matched in self.WILDCARDS_PATTERN.finditer(self.value):
            yield matched.span(), matched.group()

    def split_wildcards(self):
        """split term on wildcards
        """
        return self.WILDCARDS_PATTERN.split(self.value)

    def has_wildcard(self):
        """:return bool: True if value contains a wildcards
        """
        return any(self.iter_wildcards())

    def __str__(self):
        return self.value

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)


class Word(Term):
    pass


class Doi(Word):
    pass


class Url(Word):
    pass


class Phrase(Term):
    """A phrase term, that is a sequence of words enclose in quotes
    :param str value: the value, including the quotes. Eg. ``'"my phrase"'``
    """

    def __str__(self):
        return f'"{self.value}"'


class Regex(Term):
    """A regex term, that is a sequence of words enclose in slashes
    :param str value: the value, including without slashes
    """
    def __init__(self, value):
        super(Regex, self).__init__(value)

    def __str__(self):
        return f'/{self.value}/'


class BaseApprox(Item):
    """Base for approximations, that is fuzziness and proximity
    """
    _equality_attrs = ['term', 'slop']

    def __repr__(self):  # pragma: no cover
        return "%s(%s, %s)" % (self.__class__.__name__, self.term.__repr__(), self.slop)

    @property
    def children(self):
        return [self.term]


class Fuzzy(BaseApprox):
    """Fuzzy search on word
    :param Word term: the approximated term
    :param slop: the degree which will be converted to :py:class:`decimal.Decimal`.
    """
    def __init__(self, term, slop=None):
        super().__init__()
        self.term = term
        if slop is None:
            slop = 0.5
        self.slop = Decimal(slop).normalize()

    def __str__(self):
        return "%s~%s" % (self.term, self.slop)


class Proximity(BaseApprox):
    """Proximity search on phrase
    :param phrase: the approximated phrase
    :param slop: the degree which will be converted to :py:func:`int`.
    """
    def __init__(self, term, slop=None):
        super().__init__()
        self.term = term
        if slop is None:
            slop = 1
        self.slop = int(slop)

    def __str__(self):
        return "%s~" % self.term + ("%d" % self.slop if self.slop is not None else "")


class Boost(Item):
    """A term for boosting a value or a group there of
    :param expr: the boosted expression
    :param force: boosting force, will be converted to :py:class:`decimal.Decimal`
    """
    def __init__(self, expr, score: str):
        super().__init__()
        self.expr = expr
        if score is None:
            score = '1.0'
        self.score = Decimal(score).normalize()

    @property
    def children(self):
        """The only child is the boosted expression
        """
        return [self.expr]

    def __repr__(self):  # pragma: no cover
        return "%s(%s, %s)" % (self.__class__.__name__, self.expr.__repr__(), str(self.score))

    def __str__(self):
        return "%s^%s" % (self.expr.__str__(), str(self.score))

    def to_summa_query(self):
        return {'boost': {'query': self.expr.to_summa_query(), 'score': str(self.score)}}


class Unary(Item):
    """Parent class for unary operations
    :param a: the expression the operator applies on
    """

    def __init__(self, a):
        self.a = a

    def __str__(self):
        return "%s%s" % (self.op, self.a.__str__())

    def to_summa_query(self):
        return {'boolean': {'subqueries': [{'occur': 'should', 'query': self.a.to_summa_query()}]}}

    @property
    def children(self):
        return [self.a]


class Plus(Unary):
    """plus, unary operation
    """
    op = "+"

    def to_summa_query(self):
        return {'boolean': {'subqueries': [{'occur': 'must', 'query': self.a.to_summa_query()}]}}


class Minus(Unary):
    """The negation
    """
    op = "-"

    def to_summa_query(self):
        return {'boolean': {'subqueries': [{'occur': 'must_not', 'query': self.a.to_summa_query()}]}}
