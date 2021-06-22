from typing import (
    Optional,
    Tuple,
)

from izihawa_utils.text import camel_to_snake


class TreeVisitor:
    visitor_method_prefix = 'visit_'
    generic_visitor_method_name = 'generic_visit'

    _get_method_cache = None

    def __init__(self, ignore_nodes: Optional[Tuple] = None):
        self.ignore_nodes = ignore_nodes or tuple()

    def has_parent(self, parents, clses):
        return parents is not None and len(parents) > 0 and isinstance(parents[-1], clses)

    def has_ancestor(self, parents, clses):
        return parents is not None and len(parents) > 0 and any([isinstance(parent, clses) for parent in parents])

    def _get_method(self, node):
        if self._get_method_cache is None:
            self._get_method_cache = {}
        try:
            meth = self._get_method_cache[type(node)]
        except KeyError:
            for cls in node.__class__.mro():
                try:
                    method_name = "{}{}".format(
                        self.visitor_method_prefix,
                        camel_to_snake(cls.__name__)
                    )
                    meth = getattr(self, method_name)
                    break
                except AttributeError:
                    continue
            else:
                meth = getattr(self, self.generic_visitor_method_name)
            self._get_method_cache[type(node)] = meth
        return meth

    def visit(self, node, context, parents=None):
        """ Basic, recursive traversal of the tree. """
        parents = parents or []
        if self.ignore_nodes and isinstance(node, self.ignore_nodes):
            yield from iter([])
        method = self._get_method(node)
        yield from method(node, parents=parents, context=context)
        for child in node.children:
            yield from self.visit(child, parents=parents + [node], context=context)

    def generic_visit(self, node, context, parents=None):
        """
        Default visitor function, called if nothing matches the current node.
        """
        return iter([])  # No-op


class TreeTransformer(TreeVisitor):
    def replace_node(self, old_node, new_node, parent):
        for k, v in parent.__dict__.items():  # pragma: no branch
            if v == old_node:
                parent.__dict__[k] = new_node
                break
            elif isinstance(v, list):
                try:
                    i = v.index(old_node)
                    if new_node is None:
                        del v[i]
                    else:
                        v[i] = new_node
                    break
                except ValueError:
                    pass  # this was not the attribute containing old_node
            elif isinstance(v, tuple):
                try:
                    i = v.index(old_node)
                    v = list(v)
                    if new_node is None:
                        del v[i]
                    else:
                        v[i] = new_node
                    parent.__dict__[k] = tuple(v)
                    break
                except ValueError:
                    pass  # this was not the attribute containing old_node

    def generic_visit(self, node, context, parents=None):
        return node, False

    def visit(self, node, context, parents=None):
        """
        Recursively traverses the tree and replace nodes with the appropriate
        visitor method's return values.
        """
        if not node or (self.ignore_nodes and isinstance(node, self.ignore_nodes)):
            return node
        parents = parents or []
        method = self._get_method(node)
        new_node, is_final = method(node, context=context, parents=parents)
        if parents:
            self.replace_node(node, new_node, parents[-1])
        node = new_node
        if node is not None and not is_final:
            for child in node.children:
                self.visit(child, context=context, parents=parents + [node])
        return node
