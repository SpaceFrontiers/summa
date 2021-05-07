from aiosumma.parser import default_parser


def test_parser():
    parsed_query_1 = default_parser.parse(
        '(hemoglobin- er OR hemoglobins-a) '
        'AND -fetal AND (human to monkey - is cool) '
        'AND year:[1992 to 1994]'
    )
    assert str(parsed_query_1) == '(((hemoglobin- er OR hemoglobins-a)) AND -fetal ' \
                                  'AND ((human to monkey is cool)) AND year:[1992 TO 1994])'
    assert (str(default_parser.parse("bek OR 'kek'")) == '(bek OR "kek")')
    assert (str(default_parser.parse("bek OR 'kek")) == '(bek OR kek)')
    assert (str(default_parser.parse("bek OR a'kek")) == '(bek OR a kek)')
    assert (str(default_parser.parse("bek' OR 'kek mek'")) == '(bek " OR " kek mek)')

    assert (str(default_parser.parse("field:test")) == 'field:test')
    assert (str(default_parser.parse("field: test")) == '(field test)')
    assert (str(default_parser.parse("field : test")) == '(field test)')
    assert (str(default_parser.parse("field :test")) == '(field test)')
