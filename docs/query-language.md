# Query language

Summa accepts terms, field-qualified terms, phrases, prefixes, wildcard and regex functions, explicit
`AND`/`OR`/`NOT`, grouping, unary `+`/`-` modifiers, and vector expressions.
Whitespace between clauses is an implicit OR.

## Required and prohibited clauses

Unary `+` marks a required clause; unary `-` marks a prohibited clause. A
modifier binds to the immediately following term, phrase, prefix,
field-qualified expression, or parenthesized group, and applies within its
nearest Boolean group. A `+` or `-` followed by whitespace is ordinary text,
not a modifier. An unmodified clause is optional in an OR group and required
in an AND group. A group with required clauses matches only when all of them
match; its optional clauses add score. A prohibited clause excludes matching
documents and adds no score. A standalone prohibited clause matches every
document except its matches.

| Expression             | Matching rule                                                |
| ---------------------- | ------------------------------------------------------------ |
| `alpha beta`           | Either term                                                  |
| `+alpha beta`          | Alpha required; beta contributes if present                  |
| `+alpha +beta`         | Both terms required                                          |
| `alpha -beta`          | Alpha present and beta absent                                |
| `alpha - beta`         | Either term; the bare dash is text                           |
| `+"alpha beta" +gamma` | Adjacent phrase and gamma required                           |
| `+(alpha beta) -gamma` | Either alpha or beta, excluding gamma                        |
| `(+alpha) beta`        | The group or beta; the inner modifier stays inside its group |
| `alpha OR NOT beta`    | Explicit Boolean OR with a complement                        |

Precedence is unary modifier/complement, then `AND`, then explicit or implicit
`OR`. Parentheses bound the scope of a modifier. `NOT`/`!` is a complement and
is distinct from a prohibited `-` clause: `alpha OR NOT beta` does not become
`alpha AND NOT beta`.

Word operators are complete tokens: separate them from a following word or
modifier with whitespace; a parenthesis or quote also delimits the keyword.
`NOTHING`, `ANDROID`, `ORCHID`, field names and prefixes are not split into
operators. Symbolic `&&`, `||` and `!` keep their punctuation syntax.

## Parsing entry points

`parse` keeps the free-text fallback: input that does not parse as query
syntax (for example punctuation in natural language) becomes an OR of its
tokens. A malformed explicit modifier such as `+`, `++alpha` or `+()` is an
error rather than silently losing its modifier; a bare `+` or `-` token is
free text. `parse_strict` reports every syntax error. Configured field routing
runs before either entry point. A single unqualified term over one default
field builds a direct term query, including inside a larger Boolean query.

Compatibility: `-` previously shared `NOT`'s complement behavior. It now means
a prohibited clause in its Boolean group. Use `NOT` or `!` when an OR branch
should match a complement.

## Phrases

Quoted phrases containing multiple analyzed tokens require an indexed text
field with `token_position` or `positions`. Missing or ordinal-only positions
produce a field-specific error; Summa does not replace adjacency with AND.
Use an explicit AND for unordered terms, or rebuild the field with token
positions. Unqualified phrases keep all configured default-field branches,
including branches that analyze to one token; an unsupported multi-token
branch is an error.

## Wildcard term filters

`field:wildcard("foo*bar?")` matches complete indexed terms. `*` matches zero or
more Unicode characters; `?` matches exactly one. The pattern is lowercased,
but not tokenized or stemmed. Arguments use JSON string escaping: for example,
`field:wildcard("a\\*b")` matches the literal term `a*b`. An unqualified function
searches the default fields. Every matching document scores 1.0 per field,
regardless of how many terms matched. Bare patterns such as `field:foo*bar?` and `*suffix` are also supported. A simple
trailing-star pattern retains the existing prefix query path. Interior patterns
are consumed as one query rather than split into separate clauses.

Prefix and wildcard filters share bounded expansion and union execution and
preserve logical document IDs on RGB fields. Chunked fields are rejected.
Leading wildcards may scan many dictionary terms; excessive expansion returns
an error. See [wildcard queries](wildcard-query.md) for the limits and core API.

## Regex filters and literal punctuation

`field:regex("(19|20)[0-9]{2}")` matches entire indexed terms. Patterns are
case-sensitive and are not analyzed or lowercased. Supported syntax includes
classes/ranges, grouping, alternation and repetition; see [regex queries](regex-query.md)
for the language and resource limits. These filters share wildcard union
execution, constant scoring, RGB ID mapping and explicit expansion errors.
Malformed regex calls and dangling term escapes are errors in both parsing entry points.

Dots and apostrophes are accepted inside ordinary terms, including
`books.google.com`, `12.6` and `hill's`. A backslash quotes the next character:
`+a +user\:ed` requires both terms, with the colon inside the second term;
`tag:a\*b` searches for a literal asterisk. The parser unescapes once and applies
the field's configured tokenizer. Use a raw field when punctuation must be
preserved verbatim in the index; this syntax does not change analyzer behavior.
