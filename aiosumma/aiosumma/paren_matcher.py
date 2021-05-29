counterparts = {
    ')': '(',
    ']': '[',
    '>': '<',
}


def remove_unmatched_parens(query):
    paren_indices = {
        '(': [],
        '[': [],
        '<': [],
    }
    processed_query = []
    index_shift = 0
    for index, ch in enumerate(query):
        if ch in paren_indices:
            paren_indices[ch].append(index - index_shift)
            processed_query.append(ch)
        elif ch in counterparts:
            if len(paren_indices[counterparts[ch]]) > 0:
                paren_indices[counterparts[ch]] = paren_indices[counterparts[ch]][:-1]
                processed_query.append(ch)
            else:
                index_shift += 1
        else:
            processed_query.append(ch)
    delete_indices = []
    for paren in paren_indices:
        delete_indices.extend(paren_indices[paren])
    if delete_indices:
        delete_indices = sorted(delete_indices, reverse=True)
    for ix in delete_indices:
        del processed_query[ix]
    return ''.join(processed_query)
