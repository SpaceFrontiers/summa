# Query language

Hermes accepts terms, field-qualified terms, phrases, prefixes, explicit
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
produce a field-specific error; Hermes does not replace adjacency with AND.
Use an explicit AND for unordered terms, or rebuild the field with token
positions. Unqualified phrases keep all configured default-field branches,
including branches that analyze to one token; an unsupported multi-token
branch is an error.
