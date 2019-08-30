# -*- coding: utf-8 -*-
"""
Module to work on symbolic algebra graphs
"""

from .datatype import (SYM_ONE, SYM_ZERO, SYM_TWO, Salt, Leaf, s_add,
                       s_neg, s_sub, s_mul, s_div, s_sel, s_unary, s_pow,
                       ID_SRC, ID_ZERO, ID_ONE, ID_ADD, ID_SUB, ID_MUL, ID_DIV,
                       ID_NEG, ID_SEL, ID_POW, ID_SIN, ID_COS, ID_TAN, ID_ASIN,
                       ID_ACOS, ID_ATAN, ID_SINH, ID_COSH, ID_TANH, ID_SQRT,
                       ID_EXP, ID_LOG, ID_INV, ID_SQU, ID_PLAIN, NUM_IDS)


class SaltArray(object):
    """This class represents an array specialised for symbols in it.

    You may instantiate this list with anything in it, but for the
    specific methods to work, the containing datatypes better are other
    containers or objects of type :py:obj:`Salt <salt.datatype.Salt>`.
    Derivatives are represented as ``SaltArray`` objects.
    Some slicing functionality is included, and indexing supports
    multi-dimensional lists. Furthermore the objects are iteratable.
    """
    def __init__(self, source=None):
        """Constructor of an empty object or one based on the given source.

        :param source: The symbols to be treated as a collection. Containers
            can be nested and inhomogenious, as long as they are either
            iterable or of type :py:obj:`Salt <salt.datatype.Salt>`.
            If ``source`` is not provided (or ``None``), the object is
            initialised as an empty container.
        :type source: Iterable (nested) container of
            :py:obj:`Salt <salt.datatype.Salt>`
        """
        self.__source = [] if source is None else source

    def __len__(self):
        return len(self.__source)

    def index(self, entry):
        """Same method as the corresponding one for ``list`` objects."""
        return self.__source.index(entry)

    def append(self, data):
        """Same method as the corresponding one for ``list`` objects."""
        self.__source.append(data)

    def extend(self, data):
        """Same method as the corresponding one for ``list`` objects."""
        self.__source.extend(data)

    def __iter__(self):
        return self.__source.__iter__()

    @staticmethod
    def __get_item_of_container(container, key):
        if type(key) not in (int, slice, tuple):
            raise TypeError("list indices must be integers or slices")
        try:
            return container[key]
        except TypeError:
            res = SaltArray.__get_item_of_container(container, key[0])
            if len(key) == 1:
                return res
            else:
                return SaltArray.__get_item_of_container(res, *key[1:])

    def __getitem__(self, key):
        res = SaltArray.__get_item_of_container(self.__source, key)

        try:
            res[0]
        except TypeError:  # it's just a scalar symbol
            return res
        else:  # wrap as a SaltArray
            return SaltArray(res)

    def invalidate(self):
        """Same as :py:obj:`Salt.invalidate <salt.datatype.Salt.invalidate>`,
        just applied to the entire container, therefore slightly more
        efficient
        :return: None
        """
        SaltArray.invalidate_container(self.__source)

    @staticmethod
    def invalidate_container(container):
        """The static version of
        :py:obj:`invalidate <salt.tools.SaltArray.invalidate>`, can be applied
        to any (nested) container"""
        for elem in container:
            try:
                elem.invalidate()
            except AttributeError:
                SaltArray.invalidate_container(elem)

    def _get_value(self):
        return SaltArray.value_container(self.__source)
    value = property(_get_value)
    """Same as :py:obj:`Salt.value <salt.datatype.Salt.value>`,
    just applied to the entire container and returning a nested
    container of same shape as original, containing the float values

    :return: The values of the symbols within the container
    :rtype: ``<list <list ...<float>...>``
    """

    @staticmethod
    def value_container(container):
        """The static version of
        :py:obj:`value <salt.tools.SaltArray.value>`, can be applied
        to any (nested) container"""
        def _val(obj):
            try:
                return obj.value
            except AttributeError:
                return SaltArray.value_container(obj)

        return [_val(elem) for elem in container]

    def recalc(self):
        """Same as :py:obj:`Salt.recalc <salt.datatype.Salt.recalc>`,
        just applied to the entire container, therefore slightly more
        efficient

        :return: The values of the symbols within the container
        :rtype: ``<list <list ...<float>...>``
        """
        return SaltArray.recalc_container(self.__source)

    @staticmethod
    def recalc_container(container):
        """The static version of
        :py:obj:`recalc <salt.tools.SaltArray.recalc>`, can be applied
        to any (nested) container"""
        SaltArray.value_container(container)
        SaltArray.invalidate_container(container)
        return SaltArray.value_container(container)


def sparse_derivative(dependent, independent):
    deris = {id(x.node): {id(x.node): SYM_ONE.dup()} for x in independent}
    dependent = [y.node for y in dependent]

    def chain_rule(node, childs):
        """Derivative of node with respect to x, using inner derivatives of the
        children with respect to x. dcdx needs to be released if not used, but
        the node children of course not."""
        # pylint: disable=too-many-locals

        def _add(_, childs):  # (a+b)' = a' + b'
            return {i_id: s_add(c[0], c[1]) for i_id, c in childs.items()}

        def _sub(_, childs):  # (a-b)' = a' - b'
            return {i_id: s_sub(c[0], c[1]) for i_id, c in childs.items()}

        def _mul(node, childs):  # y' = a*b' + a'*b
            terms = {i_id: (s_mul(node.childs[0].dup(), c[1]),
                            s_mul(node.childs[1].dup(), c[0]))
                     for i_id, c in childs.items()}
            return {i_id: s_add(t[0], t[1]) for i_id, t in terms.items()}

        def _div(node, childs):  # y' = (a'-y*b')/b
            return {i_id: s_div(s_sub(c[0], s_mul(node.dup(), c[1])),
                                node.childs[1].dup())
                    for i_id, c in childs.items()}

        def _squ(node, childs):  # a²' = 2*a*a'
            fac = s_mul(SYM_TWO.dup(), node.childs[0].dup())
            res = {i_id: s_mul(fac.dup(), c[0])
                   for i_id, c in childs.items()}
            fac.release()
            return res

        def _neg(_, childs):  # (-a)' = -a'
            return {i_id: s_neg(c[0]) for i_id, c in childs.items()}

        def _sel(node, childs):  # sel(a, b)' = sel(a, b')
            res = {i_id: s_sel(node.childs[0].dup(), c[1])
                   for i_id, c in childs.items()}
            for c in childs.values():
                c[0].release()
            return res

        def _pow(node, childs):  # y' = (a**b)' = y*log(a)*b' + y*b/a*a'
            yloga = s_mul(node.dup(), s_unary(ID_LOG, node.childs[0].dup()))
            yxb = s_mul(node.dup(), node.childs[1].dup())
            yba = s_div(yxb, node.childs[0].dup())
            res = {i_id: s_add(s_mul(yloga.dup(), c[1]),
                               s_mul(yba.dup(), c[0]))
                   for i_id, c in childs.items()}
            yloga.release()
            yba.release()
            return res

        def _sin(node, childs):  # sin(a)' = cos(a)*a'
            term1 = s_unary(ID_COS, node.childs[0].dup())
            res = {i_id: s_mul(term1.dup(), c[0])
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _cos(node, childs):  # cos(a)' = -sin(a)*a'
            term1 = s_neg(s_unary(ID_SIN, node.childs[0].dup()))
            res = {i_id: s_mul(term1.dup(), c[0])
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _tan(node, childs):  # tan(a)' = a'/cos²(a)
            term1 = s_unary(ID_COS, node.childs[0].dup())
            term2 = s_unary(ID_INV, s_unary(ID_SQU, term1))
            res = {i_id: s_mul(term2.dup(), c[0])
                   for i_id, c in childs.items()}
            term2.release()
            return res

        def _asin(node, childs):  # asin(a)' = a'/sqrt(1-a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_unary(ID_SQRT, s_sub(SYM_ONE.dup(), term1))
            res = {i_id: s_div(c[0], term2.dup())
                   for i_id, c in childs.items()}
            term2.release()
            return res

        def _acos(node, childs):  # acos(a)' = -a'/sqrt(1-a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_neg(s_unary(ID_SQRT, s_sub(SYM_ONE.dup(), term1)))
            res = {i_id: s_div(c[0], term2.dup())
                   for i_id, c in childs.items()}
            term2.release()
            return res

        def _atan(node, childs):  # atan(a)' = a'/(1+a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_add(SYM_ONE.dup(), term1)
            res = {i_id: s_div(c[0], term2.dup())
                   for i_id, c in childs.items()}
            term2.release()
            return res

        def _sinh(node, childs):  # sinh(a)' = cosh(a)*a'
            term1 = s_unary(ID_COSH, node.childs[0].dup())
            res = {i_id: s_mul(term1.dup(), c[0])
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _cosh(node, childs):  # cosh(a)' = sinh(a)*a'
            term1 = s_unary(ID_SINH, node.childs[0].dup())
            res = {i_id: s_mul(term1.dup(), c[0])
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _tanh(node, childs):  # tanh(a)' = a'/cosh²(a)
            term1 = s_unary(ID_SQU, s_unary(ID_COSH, node.childs[0].dup()))
            res = {i_id: s_div(c[0], term1.dup())
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _sqrt(node, childs):  # y = sqrt(a)' = a'/(2*y)
            term1 = s_mul(SYM_TWO.dup(), node.dup())
            res = {i_id: s_div(c[0], term1.dup())
                   for i_id, c in childs.items()}
            term1.release()
            return res

        def _log(node, childs):  # log(a)' = a'/a
            return {i_id: s_div(c[0], node.childs[0].dup())
                    for i_id, c in childs.items()}

        def _exp(node, childs):  # y = exp(a)' = a' * y
            return {i_id: s_mul(c[0], node.dup())
                    for i_id, c in childs.items()}

        def _inv(node, childs):  # inv(a)' = a'/squ(a)
            term1 = s_neg(s_unary(ID_SQU, node.childs[0].dup()))
            res = {i_id: s_div(c[0], term1.dup())
                   for i_id, c in childs.items()}
            term1.release()
            return res

        rules = {ID_ADD: _add, ID_SUB: _sub, ID_MUL: _mul, ID_DIV: _div,
                 ID_SQU: _squ, ID_NEG: _neg, ID_SEL: _sel, ID_POW: _pow,
                 ID_SIN: _sin, ID_COS: _cos, ID_TAN: _tan, ID_ASIN: _asin,
                 ID_ACOS: _acos, ID_ATAN: _atan, ID_SINH: _sinh,
                 ID_COSH: _cosh, ID_TANH: _tanh, ID_SQRT: _sqrt, ID_LOG: _log,
                 ID_EXP: _exp, ID_INV: _inv}

        #  make {indep_i: [dc_j_dx_i]}
        #  insert zeros where necessary
        if len(childs) == 1:  # frequent case, much easier
            childs2 = {i_id: [deri] for i_id, deri in childs[0].items()}
        else:
            childs2 = {}
            for key in set().union(*[dcdx_i.keys() for dcdx_i in childs]):
                entry = [dcdx_i.get(key, None) for dcdx_i in childs]
                for k, e_i in enumerate(entry):
                    if e_i is None:
                        entry[k] = SYM_ZERO.dup()
                childs2[key] = entry
        return rules[node.tid](node, childs2)

    def derive(node):
        node_id = id(node)
        if node_id not in deris:
            if node.tid in (ID_ZERO, ID_ONE, ID_SRC):
                deris[node_id] = {}
            else:
                dcdx = [derive(child) for child in node.childs]
                if True in [bool(cx_i) for cx_i in dcdx]:
                    deris[node_id] = chain_rule(node, dcdx)
                else:
                    for cx_i in dcdx:
                        for cx_ij in cx_i.values():
                            cx_ij.release()
                    deris[node_id] = {}
        return deris[node_id]

    # index in independent vector as function of id
    indep_idx = {id(x.node): k for k, x in enumerate(independent)}

    result = {}
    for k, dep in enumerate(dependent):
        entry = derive(dep)
        if entry:  # node dependent on independent variables
            entry = {indep_idx[i_id]: Salt(d_node)
                     for i_id, d_node in entry.items()}
            result[k] = entry
    return result


class Derivative(SaltArray):
    """
    This class could have been implemented as a function, as it consciously
    behaves like one. However, the cleanest way to encapsulate it's (private)
    content is probably to define it as a class, prepresenting its own result.

    When deriving (right under construction), the dependent variables and their
    underlying graph are first analysed in order to avoid chewing on derivatives
    that are anyway zero. Then, the graph is again traversed recursively
    in order to obtain and pre-simplify the derivatives with respect to
    the independent variables.

    In fear of performance issues, the final more rigorous simplification
    is not included here, but might be in the future.
    """
    def __init__(self, dependent, independent):
        """Construct the result object as the symbolic derivative
        :math:`\\mathrm{d}y/\\mathrm{d}x`.

        :param dependent: The variables *y* to derive
        :type dependent: Iterable container of
          :py:obj:`Salt <salt.datatype.Salt>`
        :param independent: The variables *x* to derive with to
        :type independent: Iterable container of
          :py:obj:`Salt <salt.datatype.Salt>`
        """
        SaltArray.__init__(self)
        indep = [id(x.node) for x in independent]
        dep = [y.node for y in dependent]

        # find dependencies and already cache trivial derivatives
        self.__dependencies = self.__find_dependencies(dep, indep)
        self.__cache = dict((nid, [SYM_ONE.dup()]) for nid in indep)
        # evaluate actual derivatives
        for node in dep:
            row = [Salt(SYM_ZERO.dup()) for _ in independent]
            dydx, indep_y = self.__derive(node)
            for iid, dydxi in zip(indep_y, dydx):
                row[indep.index(iid)] = Salt(dydxi)
            self.append(row)

        # cache and dependency information might take much memory. Clean it!
        del self.__dependencies
        for nodes in self.__cache.values():
            for node in nodes:
                node.release()
        del self.__cache

    def __derive(self, node):
        nid = id(node)
        indep = self.__dependencies[nid]
        if not indep:  # independent branch of graph
            return [], []
        try:  # is result cached (nid in self.__cache)?
            result = [deri.dup() for deri in self.__cache[nid]]
        except KeyError:
            child_deris = [self.__derive(child) for child in node.childs]
            child_deris = self.__unmask(child_deris, indep)
            result = self.__chain_rule(node, child_deris)
            self.__cache[nid] = [deri.dup() for deri in result]
        return result, indep

    @staticmethod
    def __chain_rule(node, dcdx):
        """Derivative of node with respect to x, using inner derivatives of the
        children with respect to x. dcdx needs to be released if not used, but
        the node children of course not."""
        # pylint: disable=too-many-locals

        def _add(_, dcdx):  # (a+b)' = a' + b'
            return [s_add(c1, c2) for c1, c2 in zip(*dcdx)]

        def _sub(_, dcdx):  # (a-b)' = a' - b'
            return [s_sub(c1, c2) for c1, c2 in zip(*dcdx)]

        def _mul(node, dcdx):  # y' = a*b' + a'*b
            term1 = [s_mul(node.childs[0].dup(), c2) for c2 in dcdx[1]]
            term2 = [s_mul(node.childs[1].dup(), c1) for c1 in dcdx[0]]
            return [s_add(t1, t2) for t1, t2 in zip(term1, term2)]

        def _div(node, dcdx):  # y' = (a'-y*b')/b
            term1 = [s_mul(node.dup(), c2) for c2 in dcdx[1]]
            term2 = [s_sub(c1, t1) for c1, t1 in zip(dcdx[0], term1)]
            return [s_div(t2, node.childs[1].dup())
                    for t1, t2 in zip(term1, term2)]

        def _squ(node, dcdx):  # a²' = 2*a*a'
            term1 = s_mul(SYM_TWO.dup(), node.childs[0].dup())
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _neg(_, dcdx):  # (-a)' = -a'
            return [s_neg(c1) for c1 in dcdx[0]]

        def _sel(node, dcdx):  # sel(a, b)' = sel(a, b')
            res = [s_sel(node.childs[0].dup(), c1) for c1 in dcdx[1]]
            for orphan in dcdx[0]:
                orphan.release()
            return res

        def _pow(node, dcdx):  # y' = (a**b)' = y*log(a)*b' + y*b/a*a'
            term1 = s_mul(node.dup(), s_unary(ID_LOG, node.childs[0].dup()))
            term2 = s_div(s_mul(node.dup(), node.childs[1].dup()), node.childs[0].dup())
            term3 = [s_mul(term1.dup(), c2) for c2 in dcdx[1]]
            term4 = [s_mul(term2.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            term2.release()
            return [s_add(t3, t4) for t3, t4 in zip(term3, term4)]

        def _sin(node, dcdx):  # sin(a)' = cos(a)*a'
            term1 = s_unary(ID_COS, node.childs[0].dup())
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _cos(node, dcdx):  # cos(a)' = -sin(a)*a'
            term1 = s_neg(s_unary(ID_SIN, node.childs[0].dup()))
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _tan(node, dcdx):  # tan(a)' = a'/cos²(a)
            term1 = s_unary(ID_INV, s_unary(ID_SQU, s_unary(ID_COS, node.childs[0].dup())))
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _asin(node, dcdx):  # asin(a)' = a'/sqrt(1-a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_unary(ID_SQRT, s_sub(SYM_ONE.dup(), term1))
            res = [s_div(c1, term2.dup()) for c1 in dcdx[0]]
            term2.release()
            return res

        def _acos(node, dcdx):  # acos(a)' = -a'/sqrt(1-a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_neg(s_unary(ID_SQRT, s_sub(SYM_ONE.dup(), term1)))
            res = [s_div(c1, term2.dup()) for c1 in dcdx[0]]
            term2.release()
            return res

        def _atan(node, dcdx):  # atan(a)' = a'/(1+a²)
            term1 = s_unary(ID_SQU, node.childs[0].dup())
            term2 = s_add(SYM_ONE.dup(), term1)
            res = [s_div(c1, term2.dup()) for c1 in dcdx[0]]
            term2.release()
            return res

        def _sinh(node, dcdx):  # sinh(a)' = cosh(a)*a'
            term1 = s_unary(ID_COSH, node.childs[0].dup())
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _cosh(node, dcdx):  # cosh(a)' = sinh(a)*a'
            term1 = s_unary(ID_SINH, node.childs[0].dup())
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _tanh(node, dcdx):  # tanh(a)' = a'/cosh²(a)
            term1 = s_unary(ID_INV, s_unary(ID_SQU, s_unary(ID_COSH, node.childs[0].dup())))
            res = [s_mul(term1.dup(), c1) for c1 in dcdx[0]]
            term1.release()
            return res

        def _sqrt(node, dcdx):  # y = sqrt(a)' = a'/(2*y)
            term1 = s_mul(SYM_TWO.dup(), node.dup())
            result = [s_div(c1, term1.dup()) for c1 in dcdx[0]]
            term1.release()
            return result

        def _log(node, dcdx):  # log(a)' = a'/a
            return [s_div(c1, node.childs[0].dup()) for c1 in dcdx[0]]

        def _exp(node, dcdx):  # y = exp(a)' = a' * y
            return [s_mul(c1, node.dup()) for c1 in dcdx[0]]

        def _inv(node, dcdx):  # inv(a)' = a'/squ(a)
            term1 = s_neg(s_unary(ID_SQU, node.childs[0].dup()))
            result = [s_div(c1, term1.dup()) for c1 in dcdx[0]]
            term1.release()
            return result

        rules = {ID_ADD: _add, ID_SUB: _sub, ID_MUL: _mul, ID_DIV: _div,
                 ID_SQU: _squ, ID_NEG: _neg, ID_SEL: _sel, ID_POW: _pow,
                 ID_SIN: _sin, ID_COS: _cos, ID_TAN: _tan, ID_ASIN: _asin,
                 ID_ACOS: _acos, ID_ATAN: _atan, ID_SINH: _sinh,
                 ID_COSH: _cosh, ID_TANH: _tanh, ID_SQRT: _sqrt, ID_LOG: _log,
                 ID_EXP: _exp, ID_INV: _inv}
        if node.tid in (ID_ZERO, ID_ONE, ID_SRC):
            return []
        else:
            return rules[node.tid](node, dcdx)

    @staticmethod
    def __unmask(tuples, indep):
        """Take the tuples of (nodes, ind_y) and unmask them to a vector
        with elements as indep. It can be assumed that ind_y is a subset of
        indep. Assumes original nodes to be released (reused in result)"""
        return [[deris[ind_y.index(nid)] if nid in ind_y else SYM_ZERO.dup()
                 for nid in indep] for deris, ind_y in tuples]

    @staticmethod
    def __find_dependencies(dependent, indep):
        """Generate a dictionary that knows the dependency of all involved
        nodes of the independent ones. Dictionary values are a list (not a set)
        in order to define sequence reproducibly"""
        res = {}

        def _add_dep(node):
            idn = id(node)
            if idn in indep:
                res[idn] = this = [idn]
            elif node.tid == ID_PLAIN:
                res[idn] = this = []
            elif idn in res:
                this = res[idn]
            else:
                sets = [_add_dep(child) for child in node.childs]
                res[idn] = this = list(set().union(*sets))
            return this

        for dep in dependent:
            _add_dep(dep)
        return res


def dump(symbols, scope=None):
    """This class dumps valid python code that defines the given symbols.
    The code is always generated down to the leaf nodes.

    A list of string representation of the symbolic graph. Here, the entire
    graph below *symbols* is processed down to the
    :py:obj:`Leaf <salt.datatype.Leaf>` nodes. A scope can be provided
    as an argument, providing the algorithm with names of user-known
    variables. The following example::

        a, b, c, d, e = map(Leaf, range(5))
        f = (a + b) * c
        g = b * b
        h = d * e
        i = h + f
        scope = {"a": a, "b": b, "c": c, "d": d, "e": e,
                 "f": f, "g": g, "h": h, "result": i}
        print "\\n".join(dump([f,g,h,i], scope))

    will produce the following output::

        var_1 = a + b
        f = var_1 * c
        g = b ** 2
        h = d * e
        result = h + f

    Note that ``var_1`` is no variable known at user scope, but an internal
    node. It will therefore be given a generic name, as all variables that
    are not member of ``scope``. Let us emphasise that *SALT* itself
    does not hold any symbol names in the nodes. When dumping the graph,
    the user is free to call them then and there by his/her favourite pet.

    Actually, if the dumped code is to be used to be executed (somewhere else)
    later, you might want to utilise some variable groups as lists. If above
    code is to be a function with ``[a, b, c, d, e]`` as argument ``x``,
    define scope as such::

        scope = dict("(x[%d]" % i, var) for i, var in enumerate((a, b, c, d, e))}
        scope.update(f=f, g=g, h=h, result=i)

    :param symbols: The symbols for which the graph shall be dumped
    :type symbols: Iterable 1D container of
      :py:obj:`Salt <salt.datatype.Salt>`
    :param scope: A dictionary to map variable names to known symbols
    :type scope: dict(string, :py:obj:`Salt <salt.datatype.Salt>`)
    :return: A list of strings, each of them representing an assignment
        with one operator or function (representing one symbolic node)
    :rtype: list<string>

    Note that multiple variables can point to the same node, hence *SALT*
    cannot even distinguish them. If *scope* provides multiple variables
    representing the same symbol, an arbitrary name will be selected for
    generating the string representation.
    """
    representation = {ID_SRC: None, ID_ZERO: "0", ID_ONE: "1",
                      ID_ADD: "%s + %s", ID_SUB: "%s - %s",
                      ID_MUL: "%s * %s", ID_DIV: "%s / %s", ID_NEG: "-%s",
                      ID_SEL: "0 if %s <= 0 else %s : 0", ID_POW: "%s ** %s",
                      ID_SIN: "sin(%s)", ID_COS: "cos(%s)", ID_TAN: "tan(%s)",
                      ID_ASIN: "asin(%s)", ID_ACOS: "acos(%s)", ID_ATAN: "atan(%s)",
                      ID_SINH: "sinh(%s)", ID_COSH: "cosh(%s)", ID_TANH: "tanh(%s)",
                      ID_SQRT: "sqrt(%s)", ID_EXP: "exp(%s)", ID_LOG: "log(%s)",
                      ID_INV: "1/%s", ID_SQU: "%s ** 2", ID_PLAIN: "plain(%s)"}

    result = []
    _scope = {}
    used = {}
    var_next = [0]

    def _name(node):
        nid = id(node)
        if nid in _scope:
            return _scope[nid]
        else:
            var_next[0] += 1
            result = "var_%d" % var_next[0]
            _scope[nid] = result
            return result

    def _dump(node):
        nid, tid = id(node), node.tid
        if nid in used or tid == ID_SRC:
            return
        name = _name(node)
        used[nid] = name
        for child in node.childs:
            _dump(child)
        expression = representation[tid] % tuple(map(_name, node.childs))
        result.append("%s = %s" % (name, expression))

    if scope is not None:
        for name, sym in scope.items():
            try:
                nid = id(sym.node)
            except AttributeError:
                pass
            else:
                _scope[nid] = name

    for sym in symbols:
        _dump(sym.node)
    return result


def simplify(symbols):
    """This function simplifies the given symnbol or container of symbols
    **in-place**.

    In the current implementation, it applies the same simplifications as when
    creating the graph, but simultaneously removes duplicates. That is::

        a, b = Leaf(3.14159), Leaf(2.71828)
        c = (a + b) + sin(a + b)
        simplify(c)

    will simplify to::

        x = a + b
        c = x + sin(x)

    This kind of simplification has a great impact on automatically generated
    derivatives, as the chain rule leaves a lot of common terms for the
    derivatives of different independent variables. Not all of them
    can be avoided while generating the derivatives.

    Naturally, duplicate nodes (that is: same type and same child nodes)
    can only be detected if they are under the symbolic graph reachable by
    the given symbols::

        c = sin(a+b)
        d = cos(a+b)
        simplify(c)

    This code would not be able to detect existance of *a + b* as duplicate
    somewhere else in the graph - another consequence of avoiding bidirectional
    linking, sorry - not.

    :param symbols: A single symbol or a container of symbols to be simplified.
        Containers can be nested and inhomogenious, as long as they are either
        iterable or of type :py:obj:`Salt <salt.datatype.Salt>`
    :type symbols: Iterable container of
          :py:obj:`Salt <salt.datatype.Salt>`
    :return: Number of duplicates found
    :rtype: int
    """
    explicits = {ID_ADD: s_add, ID_SUB: s_sub, ID_MUL: s_mul, ID_DIV: s_div,
                 ID_NEG: s_neg, ID_SEL: s_sel, ID_POW: s_pow}
    used = [{} for _ in range(NUM_IDS)]
    duplicates_found = [0]

    def _simplify(node):
        tid = node.tid
        if tid in (ID_SRC, ID_ZERO, ID_ONE):
            return node.dup()
        childs = map(_simplify, node.childs)
        try:
            result = explicits[tid](*childs)
        except KeyError:
            result = s_unary(tid, *childs)

        # is this a duplicate
        used_now = used[result.tid]
        key = tuple(map(id, result.childs))
        if tid in (ID_ADD, ID_MUL):
            key = tuple(sorted(key))
        if key in used_now:
            result.release()
            duplicates_found[0] += 1
            return used_now[key].dup()
        else:
            used_now[key] = result
            return result

    def _simplify_container(obj):
        try:  # is it a single node?
            node = obj.node
        except AttributeError:  # no
            for entry in obj:
                _simplify_container(entry)
        else:  # yes
            node = _simplify(node)
            obj.node.release()
            obj.node = node

    _simplify_container(symbols)
    return duplicates_found[0]


def empanada(func, inp, dim_out=1):
    """This function is described :ref:`here <empanada_empanadina>`.

    :param func: The function to be embedded, taking a list of input
        variables as argument - to be consistent with :code:`inp`,
        and returning a list of values with its dimensionality given by
        :code:`dim_out`, as well as the Jacobian :math:`J` as the derivative
        matrix of output variables with respect to input variables.
    :type func: f: list<float> -> (list<float>, list<list<float>>)
    :param inp: The list of input symbols that will be linked to the function
        input arguments
    :type inp: list<:py:obj:`Salt <salt.datatype.Salt>`>
    :return: A list of symbols linked to the return values of
        :code:`func`, with the first derivative being represented by :math:`J`
    :rtype: list<:py:obj:`Salt <salt.datatype.Salt>`>
    """
    output = [Leaf() for _ in range(dim_out)]
    jacobian = [[Leaf() for _ in inp] for _ in range(dim_out)]
    valid = [False]
    result = [entry - entry.plain() for entry in inp]
    result = [sum(j_ij * x_j for j_ij, x_j in zip(ji, result))
              for ji in jacobian]
    result = [out + entry for out, entry in zip(output, result)]

    def _invalidate():
        if valid[0]:
            valid[0] = False
            for entry in inp:
                entry.invalidate()

    def _get_value():
        if valid[0]:
            return
        input_v = [x.value for x in inp]
        output_v, jacobian_v = func(input_v)
        for out, out_v in zip(output, output_v):
            out.value = out_v
        for jac, jac_v in zip(jacobian, jacobian_v):
            for jac_i, jac_v_i in zip(jac, jac_v):
                jac_i.value = jac_v_i
        valid[0] = True

    for symbol in output:
        symbol.node.set_hooks(_get_value, _invalidate)
    for vector in jacobian:
        for symbol in vector:
            symbol.node.set_hooks(_get_value, _invalidate)
    return result


def empanadina(func, inp):
    """This function is described :ref:`here <empanada_empanadina>` and
    is very similar to :py:obj:`empanada <salt.datatype.empanada>`, just
    reduced for scalar usage.

    :param func: The function to be embedded, taking the input
        variable as argument, and returning the function value and the
        derivative of it with respect to the input variable.
    :type func: f: float -> (float, float)
    :param inp: The input symbol that will be linked to the function
        input argument
    :type inp: :py:obj:`Salt <salt.datatype.Salt>`
    :return: The symbol linked to the return value of
        :code:`func`, with the first derivative being represented by :math:`J`
    :rtype: :py:obj:`Salt <salt.datatype.Salt>`
    """
    def _func(inp):
        out, jac = func(inp[0])
        return [out], [[jac]]

    return empanada(_func, [inp])[0]
