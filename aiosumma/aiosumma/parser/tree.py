# -*- coding: utf-8 -*-
"""Elements that will constitute the parse tree of a query.
You may use these items to build a tree representing a query,
or get a tree as the result of parsing a query string.
"""
import re
from decimal import Decimal

from aiosumma.parser.errors import UnsupportedQueryError

_MARKER = object()
NON_DEFAULT_QUOTES = r''''“”‘«»„`'''
QUOTES = NON_DEFAULT_QUOTES + '"'
QUOTE_RE = re.compile(f'[{QUOTES}]')
NON_DEFAULT_QUOTE_RE = re.compile(f'[{NON_DEFAULT_QUOTES}]')


class Item(object):
    """Base class for all items that compose the parse tree.
    An item is a part of a request.
    """

    # /!\ Note on Item (and subclasses) __magic__ methods: /!\
    #
    # Since we're dealing with recursive structures, we must avoid using
    # the builtin helper methods when dealing with nested objects in
    # __magic__ methods.
    #
    # As the helper usually calls the relevant method, we end up with two
    # function calls instead of one, and end up hitting python's max recursion
    # limit twice as fast!
    #
    # This is why we're calling c.__repr__ instead of repr(c) in the __repr__
    # method. Same thing applies for all magic methods (__str__, __eq__, and any
    # other we might add in the future).

    _equality_attrs = []

    @property
    def children(self):
        """As base of a tree structure, an item may have children"""
        # empty by default
        return []

    def __init__(self, final=False):
        self.final = final

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


class SearchField(Item):
    """Indicate wich field the search expression operates on
    eg: *desc* in ``desc:(this OR that)``
    :param str name: name of the field
    :param expr: the searched expression
    """
    _equality_attrs = ['name']

    def __init__(self, name, expr, final=False):
        super().__init__(final=final)
        self.name = name
        self.expr = expr

    def __str__(self):
        return self.name + ":" + self.expr.__str__()

    def __repr__(self):
        return "SearchField(%r, %s)" % (self.name, self.expr.__repr__())

    def json(self):
        if isinstance(self.expr, Range):
            return {'range': {'field': self.name, 'value': self.expr.partial_json()}}
        elif isinstance(self.expr, Word):
            return {'term': {'field': self.name, 'value': self.expr.value}}
        elif isinstance(self.expr, Phrase):
            return {'phrase': {'field': self.name, 'value': self.expr.value}}
        elif isinstance(self.expr, Regex):
            return {'regex': {'field': self.name, 'value': self.expr.value}}
        else:
            raise UnsupportedQueryError(error=f'{self.expr} in search field `{self.name}`')

    @property
    def children(self):
        """the only child is the expression"""
        return [self.expr]


class Group(Item):
    """
    Parent class for binary operations are binary operation used to join expressions,
    like OR and AND
    :param operands: expressions to apply operation on
    """
    def __init__(self, *operands, final=False):
        super().__init__(final=final)
        self.operands = operands

    def __str__(self):
        return f'({" ".join(str(o) for o in self.operands)})'

    def json(self):
        subqueries = []
        for operand in self.operands:
            if isinstance(operand, Plus):
                subqueries.append({'occur': 'must', 'query': operand.a.json()})
            elif isinstance(operand, Minus):
                subqueries.append({'occur': 'must_not', 'query': operand.a.json()})
            else:
                subqueries.append({'occur': 'should', 'query': operand.json()})
        return {'bool': {'subqueries': subqueries}}

    @property
    def children(self):
        """children are left and right expressions
        """
        return self.operands


class Range(Item):
    """A Range
    :param left: left bound
    :param right: right bound
    :param bool including_left: wether left bound is included
    :param bool including_right: wether right bound is included
    """

    LEFT_CHAR = {True: '[', False: '{'}
    RIGHT_CHAR = {True: ']', False: '}'}

    def __init__(self, left, right, including_left=True, including_right=True, final=False):
        super().__init__(final=final)
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

    def partial_json(self):
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

    def __init__(self, value, final=False):
        super().__init__(final=final)
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
        return "%s(%r)" % (self.__class__.__name__, str(self))

    def json(self):
        return {'match': {'value': str(self)}}


class Word(Term):
    """A single word term
    :param str value: the value
    """
    def __init__(self, value, final=False):
        super().__init__(value, final=final)


class Phrase(Term):
    """A phrase term, that is a sequence of words enclose in quotes
    :param str value: the value, including the quotes. Eg. ``'"my phrase"'``
    """
    def __init__(self, value, final=False):
        super().__init__(value, final=final)

    def __str__(self):
        return f'"{self.value}"'


class Regex(Term):
    """A regex term, that is a sequence of words enclose in slashes
    :param str value: the value, including the slashes. Eg. ``'/my regex/'``
    """
    def __init__(self, value, final=False):
        super(Regex, self).__init__(value, final=final)

    def __str__(self):
        return f'/{self.value}/'


class Doi(Word):
    """A DOI term
    """
    def __init__(self, value, final=False):
        super().__init__(value, final=final)


class Url(Word):
    """A URL term
    """
    def __init__(self, value, final=False):
        super().__init__(value, final=final)


class BaseApprox(Item):
    """Base for approximations, that is fuzziness and proximity
    """
    _equality_attrs = ['term', 'degree']

    def __repr__(self):  # pragma: no cover
        return "%s(%s, %s)" % (self.__class__.__name__, self.term.__repr__(), self.degree)

    @property
    def children(self):
        return [self.term]


class Fuzzy(BaseApprox):
    """Fuzzy search on word
    :param Word term: the approximated term
    :param degree: the degree which will be converted to :py:class:`decimal.Decimal`.
    """
    def __init__(self, term, degree=None, final=False):
        super().__init__(final=final)
        self.term = term
        if degree is None:
            degree = 0.5
        self.degree = Decimal(degree).normalize()

    def __str__(self):
        return "%s~%s" % (self.term, self.degree)


class Proximity(BaseApprox):
    """Proximity search on phrase
    :param Phrase term: the approximated phrase
    :param degree: the degree which will be converted to :py:func:`int`.
    """
    def __init__(self, term, degree=None, final=False):
        super().__init__(final=final)
        self.term = term
        if degree is None:
            degree = 1
        self.degree = int(degree)

    def __str__(self):
        return "%s~" % self.term + ("%d" % self.degree if self.degree is not None else "")


class Boost(Item):
    """A term for boosting a value or a group there of
    :param expr: the boosted expression
    :param force: boosting force, will be converted to :py:class:`decimal.Decimal`
    """
    def __init__(self, expr, score, final=False):
        super().__init__(final=final)
        self.expr = expr
        if score is None:
            score = 1.0
        self.score = Decimal(score).normalize()

    @property
    def children(self):
        """The only child is the boosted expression
        """
        return [self.expr]

    def __repr__(self):  # pragma: no cover
        return "%s(%s, %s)" % (self.__class__.__name__, self.expr.__repr__(), self.score)

    def __str__(self):
        return "%s^%s" % (self.expr.__str__(), self.score)

    def json(self):
        return {'boost': {'query': self.expr.json(), 'score': str(self.score)}}


class Unary(Item):
    """Parent class for unary operations
    :param a: the expression the operator applies on
    """

    def __init__(self, a, final=False):
        super().__init__(final=final)
        self.a = a

    def __str__(self):
        return "%s%s" % (self.op, self.a.__str__())

    def json(self):
        return {'bool': {'subqueries': [{'occur': 'should', 'query': self.a.json()}]}}

    @property
    def children(self):
        return [self.a]


class Plus(Unary):
    """plus, unary operation
    """
    op = "+"

    def json(self):
        return {'bool': {'subqueries': [{'occur': 'must', 'query': self.a.json()}]}}


class Minus(Unary):
    """The negation
    """
    op = "-"

    def json(self):
        return {'bool': {'subqueries': [{'occur': 'must_not', 'query': self.a.json()}]}}
