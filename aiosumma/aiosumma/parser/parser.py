# -*- coding: utf-8 -*-
import re

import ply.lex as lex
import ply.yacc as yacc
from izihawa_nlptools.regex import (
    DOI_REGEX_TEXT,
    URL_REGEX_TEXT,
)

from aiosumma.parser.elements import (
    QUOTE_RE,
    QUOTES,
    Boost,
    Doi,
    Fuzzy,
    Group,
    Minus,
    Phrase,
    Plus,
    Proximity,
    Range,
    Regex,
    SearchField,
    Url,
    Word,
)
from aiosumma.parser.errors import ParseError

reserved = {
    'TO': 'TO',
    'to': 'TO',
}


# tokens of our grammar
tokens = (
    ['URL',
     'DOI',
     'TERM',
     'PHRASE',
     'REGEX',
     'APPROX',
     'BOOST',
     'MINUS',
     'PLUS',
     'COLUMN',
     'LPAREN',
     'RPAREN',
     'LBRACKET',
     'RBRACKET'
     ] + sorted(set(reserved.values()))
)


# text of some simple tokens
t_PLUS = r'\+(?=\S)'
t_MINUS = r'\-(?=\S)'
t_COLUMN = r'(?<=\S):(?=\S)'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'(\[|\{)'
t_RBRACKET = r'(\]|\})'

# precedence rules
precedence = (
    ('nonassoc', 'MINUS',),
    ('nonassoc', 'PLUS',),
    ('nonassoc', 'APPROX'),
    ('nonassoc', 'BOOST'),
    ('nonassoc', 'LPAREN', 'RPAREN'),
    ('nonassoc', 'LBRACKET', 'TO', 'RBRACKET'),
    ('nonassoc', 'REGEX'),
    ('nonassoc', 'PHRASE'),
    ('nonassoc', 'TERM'),
)

# term

# this is a wide catching expression, to also include date math.
# Inspired by the original lucene parser:
# https://github.com/apache/lucene-solr/blob/master/lucene/queryparser/src/java/org/apache/lucene/queryparser/surround/parser/QueryParser.jj#L189
# We do allow the wildcards operators ('*' and '?') as our parser doesn't deal with them.

URL_RE = fr'''
(?P<url>
  {URL_REGEX_TEXT}
)
'''

DOI_RE = fr'''
(?P<doi>
  {DOI_REGEX_TEXT}
)
'''

TERM_RE = fr'''
(?P<term>  # group term
  (?:
    [^\s:^~(){{}}[\]/,{QUOTES}+\-\\] # first char is not a space neither some char which have meanings
                               # note: escape of "-" and "]"
                               #       and doubling of "{{}}" (because we use format)
    |                          # but
    \\.                        # we can start with an escaped character
  )
  ([^\s:^\\~(){{}}[\]{QUOTES}]        # following chars
    |                          # OR
    \\.                        # an escaped cha
  )*
)
'''
# phrase
PHRASE_RE = fr'''
(?P<phrase>  # phrase
  [{QUOTES}]  # opening quote
  (?:        # repeating
    [^\\{QUOTES}]   # - a char which is not escape or end of phrase
    |        # OR
    \\.      # - an escaped char
  )*
  [{QUOTES}]  # closing quote
)'''

# r'(?P<phrase>"(?:[^\\"]|\\"|\\[^"])*")' # this is quite complicated to handle \"
# modifiers after term or phrase
APPROX_RE = r'~(?P<slop>[0-9.]+)?'
BOOST_RE = r'\^(?P<score>[0-9.]+)?'

# regex
REGEX_RE = r'''
(?P<regex>  # regex
  /         # open slash
  (?:       # repeating
    [^\\/]  # - a char which is not escape or end of regex
    |       # OR
    \\.     # an escaped char
  )*
  /         # closing slash
)'''


def t_IGNORE_HANGING_SIGNS(t):
    r"""\s+[\+-]\s+"""
    pass


def t_IGNORE_MAD_COLUMNS(t):
    r"""\s+:|:\s+|\s+:\s+"""
    pass


def t_SEPARATOR(t):
    r"""\s+"""
    pass  # discard separators


@lex.TOKEN(URL_RE)
def t_URL(t):
    t.value = Url(t.value)
    return t


@lex.TOKEN(DOI_RE)
def t_DOI(t):
    doi_r = re.match(DOI_RE, t.value, re.VERBOSE)
    t.value = Doi((doi_r[2] + '/' + doi_r[3]).lower())
    return t


@lex.TOKEN(TERM_RE)
def t_TERM(t):
    # check if it is not a reserved term (an operation)
    t.type = reserved.get(t.value, 'TERM')
    # it's not, make it a Word
    if t.type == 'TERM':
        m = re.match(TERM_RE, t.value, re.VERBOSE)
        value = m.group("term")
        t.value = Word(value)
    return t


@lex.TOKEN(PHRASE_RE)
def t_PHRASE(t):
    m = re.match(PHRASE_RE, t.value, re.VERBOSE)
    value = m.group("phrase")
    value = QUOTE_RE.sub('"', value).strip('"')
    t.value = Phrase(value)
    return t


@lex.TOKEN(REGEX_RE)
def t_REGEX(t):
    m = re.match(REGEX_RE, t.value, re.VERBOSE)
    value = m.group("regex")
    t.value = Regex(value.strip('/').replace('\\/', '/'))
    return t


@lex.TOKEN(APPROX_RE)
def t_APPROX(t):
    m = re.match(APPROX_RE, t.value)
    t.value = m.group("slop")
    return t


@lex.TOKEN(BOOST_RE)
def t_BOOST(t):
    m = re.match(BOOST_RE, t.value)
    t.value = m.group("score")
    return t


# Error handling rule FIXME
def t_error(t):  # pragma: no cover
    t.lexer.skip(1)


lexer = lex.lex()


def p_expression_implicit(p):
    """expression : expression expression"""
    if isinstance(p[2], Group):
        p[0] = Group(p[1], *p[2].operands)
    else:
        p[0] = Group(p[1], p[2])


def p_expression_plus(p):
    """unary_expression : PLUS unary_expression"""
    p[0] = Plus(p[2])


def p_expression_minus(p):
    """unary_expression : MINUS unary_expression"""
    p[0] = Minus(p[2])


def p_expression_unary(p):
    """expression : unary_expression"""
    p[0] = p[1]


def p_grouping(p):
    """unary_expression : LPAREN expression RPAREN"""
    if isinstance(p[2], Group):
        p[0] = p[2]
    else:
        p[0] = Group(p[2])


def p_range(p):
    """unary_expression : LBRACKET phrase_or_term TO phrase_or_term RBRACKET"""
    including_left = p[1] == "["
    including_right = p[5] == "]"
    p[0] = Range(p[2], p[4], including_left, including_right)


def p_field_search(p):
    """unary_expression : TERM COLUMN unary_expression"""
    p[0] = SearchField(p[1].value.lower(), p[3])


def p_quoting(p):
    """unary_expression : PHRASE"""
    p[0] = p[1]


def p_proximity(p):
    """unary_expression : PHRASE APPROX"""
    p[0] = Proximity(p[1], p[2])


def p_boosting(p):
    """expression : expression BOOST"""
    p[0] = Boost(p[1], p[2])


def p_terms(p):
    """unary_expression : URL
                        | DOI
                        | TERM"""
    p[0] = p[1]


def p_fuzzy(p):
    """unary_expression : TERM APPROX"""
    p[0] = Fuzzy(p[1], p[2])


def p_regex(p):
    """unary_expression : REGEX"""
    p[0] = p[1]


# handling a special case, TO is reserved only in range
def p_to_as_term(p):
    """unary_expression : TO"""
    p[0] = Word(p[1])


def p_phrase_or_term(p):
    """phrase_or_term : TERM
                      | PHRASE"""
    p[0] = p[1]


# Error rule for syntax errors
def p_error(p):
    if p is None:
        error = "unexpected end of expression (maybe due to unmatched parenthesis)"
        pos = "the end"
    else:
        error = "unexpected  '%s'" % p.value
        pos = "position %d" % p.lexpos
    raise ParseError(error="Syntax error in input : %s at %s!" % (error, pos))


default_parser = yacc.yacc()
"""This is the parser generated by PLY
"""
