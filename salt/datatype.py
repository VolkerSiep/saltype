# -*- coding: utf-8 -*-
"""
For a detailed documentation, see the html project documentation
"""
import math
from numpy import float64

(ID_SRC, ID_ZERO, ID_ONE, ID_ADD, ID_SUB, ID_MUL, ID_DIV, ID_NEG,
 ID_SEL, ID_POW, ID_SIN, ID_COS, ID_TAN, ID_ASIN, ID_ACOS, ID_ATAN,
 ID_SINH, ID_COSH, ID_TANH, ID_SQRT, ID_EXP, ID_LOG,
 ID_INV, ID_SQU, ID_PLAIN, NUM_IDS) = range(26)

_EFUNC = [None, None, None,
          lambda c: c[0] + c[1],
          lambda c: c[0] - c[1],
          lambda c: c[0] * c[1],
          lambda c: c[0] / c[1],
          lambda c: -c[0],
          lambda c: c[1] if c[0] > 0 else 0.0,
          lambda c: c[0] ** c[1],
          lambda c: math.sin(c[0]),
          lambda c: math.cos(c[0]),
          lambda c: math.tan(c[0]),
          lambda c: math.asin(c[0]),
          lambda c: math.acos(c[0]),
          lambda c: math.atan(c[0]),
          lambda c: math.sinh(c[0]),
          lambda c: math.cosh(c[0]),
          lambda c: math.tanh(c[0]),
          lambda c: math.sqrt(c[0]),
          lambda c: math.exp(c[0]),
          lambda c: math.log(c[0]),
          lambda c: 1.0 / c[0],
          lambda c: c[0] * c[0],
          lambda c: c[0]]


class Node(object):
    """This base-class represents a node in the Symbolic graph."""

    # pylint: disable=method-hidden, too-many-instance-attributes
    def __init__(self, typeId):
        """Initialises the Base class with a type id, no child nodes,
        and reference counter one. The value is set to invalid (NONE)"""
        self.tid = typeId
        self.childs = []
        self.ref_count = 1
        self.value = None
        self._invalidate = lambda: None  # hook functions
        self._get_value = lambda: None

    def release(self):
        """Reference counting functionality, decreasing counter by one
        and releasing children if zero. Assuming the node itself is to go
        out of scope soon after."""
        self.ref_count -= 1
        if self.ref_count == 0:
            for chi in self.childs:
                chi.release()

    def dup(self):
        """Increases the reference counter by one and returns itself for
        convenience"""
        self.ref_count += 1
        return self

    def get_value(self):
        """Assures evaluation of node's value and returns it"""
        raise NotImplementedError("Virtual function call in %s" % type(self))

    def invalidate(self):
        """Method to invalidate own value and invoke same method on children.
        Not to be implemented by always valid nodes (Zero, One and Leaf)"""
        raise NotImplementedError("Virtual function call in %s" % type(self))

    def set_hooks(self, get_value_hook=None, invalidate_hook=None):
        """Sets functions to be called if this particular node is queried
        for a value (:code:`get_value_hook`) or invalidated
        (:code:`invalidate_hook`). This is used for the
        :py:obj:`empanada <salt.tools.empanada>` and
        :py:obj:`empanadina <salt.tools.empanadina>` implementation.
        """
        if get_value_hook is not None:
            def _get_value():
                get_value_hook()
                return self._get_value()
            self._get_value = self.get_value
            self.get_value = _get_value

        if invalidate_hook is not None:
            def _invalidate():
                invalidate_hook()
                self._invalidate()
            self._invalidate = self.invalidate
            self.invalidate = _invalidate


class SymSrc(Node):
    """Subclass to represent independent nodes"""

    # pylint: disable=method-hidden
    def __init__(self, tid, value):
        """Initialise node with type id and value. As these are not direct
        user objects, the code relies on that ID_ZERO and ID_ONE are
        initialised with 0.0 and 1.0 respectively"""
        Node.__init__(self, tid)
        self.value = value

    def get_value(self):
        return self.value

    def invalidate(self):
        pass


class SymDep(Node):
    """Subclass to represent dependent nodes, i.e. such with children"""

    # pylint: disable=method-hidden
    def __init__(self, tid, *childs):
        Node.__init__(self, tid)
        self.childs = childs

    def _eval(self):
        childs = [ch.get_value() for ch in self.childs]
        self.value = _EFUNC[self.tid](childs)

    def get_value(self):
        if self.value is None:
            self._eval()
        return self.value

    def invalidate(self):
        if self.value is not None:
            self.value = None
            for chi in self.childs:
                chi.invalidate()


SYM_ZERO = SymSrc(ID_ZERO, 0.0)
SYM_ONE = SymSrc(ID_ONE, 1.0)
SYM_TWO = SymSrc(ID_SRC, 2.0)


def _fc(node, which=0):  # take child, release node itself
    res = node.childs[which].dup()
    node.release()
    return res


def s_neg(child):
    """Generate node to represent negation operator"""
    if child.tid == ID_ZERO:  # -0 = 0
        res = child
    elif child.tid == ID_NEG:  # -(-x) = x
        res = _fc(child)
    else:
        res = SymDep(ID_NEG, child)
    return res


def s_add(first, second):
    """Generate node to represent addition operator"""
    if first.tid == ID_ZERO:  # 0 + x = x
        first.release()
        res = second
    elif second.tid == ID_ZERO:  # x + 0 = x
        second.release()
        res = first
    elif first.tid == ID_NEG and second.tid == ID_NEG:  # -x + -y = -(x + y)
        res = s_neg(s_add(_fc(first), _fc(second)))
    elif first.tid == ID_NEG:  # -x + y = y - x
        res = s_sub(second, _fc(first))
    elif second.tid == ID_NEG:  # x + -y = x - y
        res = s_sub(first, _fc(second))
    elif first.tid == ID_LOG and second.tid == ID_LOG:  # log(x) + log(y) = log(x*y)
        res = s_unary(ID_LOG, s_mul(_fc(first), _fc(second)))
    else:
        res = SymDep(ID_ADD, first, second)
    return res


def s_sub(first, second):
    """Generate node to represent subtraction operator"""
    if first.tid == ID_ZERO:  # 0 - x = -x
        first.release()
        res = s_neg(second)
    elif second.tid == ID_ZERO:  # x - 0 = x
        second.release()
        res = first
    elif first.tid == ID_NEG and second.tid == ID_NEG:  # -x - -y = y - x
        res = s_sub(_fc(second), _fc(first))
    elif first.tid == ID_NEG:  # -x - y = -(x + y)
        res = s_neg(s_add(_fc(first), second))
    elif second.tid == ID_NEG:  # x - -y = x + y
        res = s_add(first, _fc(second))
    elif first.tid == ID_LOG and second.tid == ID_LOG:  # log(x) - log(y) = log(x/y)
        res = s_unary(ID_LOG, s_div(_fc(first), _fc(second)))
    elif id(first) == id(second):  # x - x = 0
        first.release()
        second.release()
        return SYM_ZERO.dup()
    else:
        res = SymDep(ID_SUB, first, second)
    return res


def s_mul(first, second):
    """Generate node to represent multiplication operator"""
    if first.tid == ID_ZERO or second.tid == ID_ONE:  # 0*x = 0, x*1 = x
        second.release()
        res = first
    elif first.tid == ID_ONE or second.tid == ID_ZERO:  # x*0 = 0, 1*x = x
        first.release()
        res = second
    elif first.tid == ID_NEG and second.tid == ID_NEG:  # (-x)*(-y) = x*y
        res = s_mul(_fc(first), _fc(second))
    elif first.tid == ID_NEG:  # (-x)*y = -(x*y)
        res = s_neg(s_mul(_fc(first), second))
    elif second.tid == ID_NEG:  # x*(-y) = -(x*y)
        res = s_neg(s_mul(first, _fc(second)))
    elif first.tid == ID_INV and second.tid == ID_INV:  # 1/x * 1/y = 1/(x*y)
        res = s_unary(ID_INV, s_mul(_fc(first), _fc(second)))
    elif first.tid == ID_INV:  # 1/x * y = y/x
        res = s_div(second, _fc(first))
    elif second.tid == ID_INV:  # x * 1/y = x/y
        res = s_div(first, _fc(second))
    elif first.tid == ID_EXP and second.tid == ID_EXP:  # exp(x)*exp(y) = exp(x+y)
        res = s_unary(ID_EXP, s_add(_fc(first), _fc(second)))
    elif first.tid == ID_SQU and second.tid == ID_SQU:  # x² * y² = (x*y)²
        res = s_unary(ID_SQU, s_mul(_fc(first), _fc(second)))
    elif id(first) == id(second):  # x*x = x²
        second.release()
        res = s_unary(ID_SQU, first)
    else:
        res = SymDep(ID_MUL, first, second)
    return res


def s_div(first, second):
    """Generate node to represent division operator"""
    if first.tid == ID_ZERO or second.tid == ID_ONE:  # 0/y = 0, x/1 = x
        second.release()
        res = first
    elif first.tid == ID_ONE:  # 1/x = inv(x)
        first.release()
        res = s_unary(ID_INV, second)
    elif first.tid == ID_NEG and second.tid == ID_NEG:  # (-x)/(-y) = x/y
        res = s_div(_fc(first), _fc(second))
    elif first.tid == ID_NEG:  # (-x)/y = -(x/y)
        res = s_neg(s_div(_fc(first), second))
    elif second.tid == ID_NEG:  # x/(-y) = -(x/y)
        res = s_neg(s_div(first, _fc(second)))
    elif first.tid == ID_INV and second.tid == ID_INV:  # (1/x)/(1/y) = y/x
        res = s_div(_fc(second), _fc(first))
    elif first.tid == ID_INV:  # (1/x)/y = 1/(x*y)
        res = s_unary(ID_INV, s_mul(_fc(first), second))
    elif second.tid == ID_INV:  # x/(1/y) = x*y
        res = s_mul(first, _fc(second))
    elif first.tid == ID_EXP and second.tid == ID_EXP:  # exp(x)/exp(y) = exp(x-y)
        res = s_unary(ID_EXP, s_sub(_fc(first), _fc(second)))
    elif id(first) == id(second):  # x/x = 1
        first.release()
        second.release()
        return SYM_ONE.dup()
    else:
        res = SymDep(ID_DIV, first, second)
    return res


def s_sel(first, second):
    """Generate node to represent selection operator"""
    if first.tid == ID_ZERO:  # sel(0, x) = 0
        second.release()
        res = first
    elif first.tid in [ID_ONE, ID_EXP]:  # sel(1, y) = sel(exp(x), y) = y
        first.release()
        res = second
    elif first.tid in [ID_SQRT, ID_SQU, ID_INV]:  # sel(sqrt(x), y) = sel(x, y)
        res = s_sel(_fc(first), second)
    else:
        res = SymDep(ID_SEL, first, second)
    return res


def s_pow(first, second):
    """Generate node to represent restructuredtext toctreepower operator"""
    if first.tid == ID_ONE or second.tid == ID_ONE:  # 1**x = 1, x**1 = x
        second.release()
        res = first
    elif second.tid == ID_ZERO:  # x**0 = 1, well: 0**0 dubious
        first.release()
        second.release()
        return SYM_ONE.dup()
    elif first.tid == ID_ZERO:  # 0**x = 0
        second.release()
        res = first
    elif first.tid == ID_EXP:  # exp(x)**y = exp(x*y)
        res = s_unary(ID_EXP, s_mul(_fc(first), second))
    else:
        res = SymDep(ID_POW, first, second)
    return res


def s_unary(tid, arg):
    """Generate node to represent unary functions"""
    def _si_sin(arg):
        res = None
        if arg.tid == ID_ZERO:  # sin(0) = 0
            res = arg
        if arg.tid == ID_NEG:  # sin(-x) = -sin(x)
            res = s_neg(s_unary(ID_SIN, _fc(arg)))
        elif arg.tid == ID_ASIN:  # sin(asin(x)) = x
            res = _fc(arg)
        elif arg.tid == ID_ACOS:  # sin(acos(x)) = sqrt(1-x²)
            tmp = s_sub(SYM_ONE.dup(), s_unary(ID_SQU, _fc(arg)))
            res = s_unary(ID_SQRT, tmp)
        elif arg.tid == ID_ATAN:  # sin(atan(x)) = x/sqrt(1+x²)
            tmp1 = _fc(arg)
            tmp2 = s_add(SYM_ONE.dup(), s_unary(ID_SQU, tmp1))
            res = s_div(tmp1.dup(), s_unary(ID_SQRT, tmp2))
        return res

    def _si_cos(arg):
        res = None
        if arg.tid == ID_ZERO:  # cos(0) = 1
            arg.release()
            res = SYM_ONE.dup()
        if arg.tid == ID_NEG:  # cos(-x) = cos(x)
            res = s_unary(ID_COS, _fc(arg))
        elif arg.tid == ID_ASIN:  # cos(asin(x)) = sqrt(1-x²)
            tmp = s_sub(SYM_ONE.dup(), s_unary(ID_SQU, _fc(arg)))
            res = s_unary(ID_SQRT, tmp)
        elif arg.tid == ID_ACOS:  # cos(acos(x)) = x
            res = _fc(arg)
        elif arg.tid == ID_ATAN:  # cos(atan(x)) = 1/sqrt(1+x²)
            tmp1 = _fc(arg)
            tmp2 = s_add(SYM_ONE.dup(), s_unary(ID_SQU, tmp1))
            res = s_div(SYM_ONE.dup(), s_unary(ID_SQRT, tmp2))
        return res

    def _si_tan(arg):
        res = None
        if arg.tid == ID_ZERO:  # tan(0) = 0
            res = arg
        if arg.tid == ID_NEG:  # tan(-x) = -tan(x)
            res = s_neg(s_unary(ID_TAN, _fc(arg)))
        elif arg.tid == ID_ASIN:  # tan(asin(x)) = x/sqrt(1-x²)
            tmp1 = _fc(arg)
            tmp2 = s_sub(SYM_ONE.dup(), s_unary(ID_SQU, tmp1))
            res = s_div(tmp1.dup(), s_unary(ID_SQRT, tmp2))
        elif arg.tid == ID_ACOS:  # tan(acos(x)) = sqrt(1-x²)/x
            tmp1 = _fc(arg)
            tmp2 = s_sub(SYM_ONE.dup(), s_unary(ID_SQU, tmp1))
            res = s_div(s_unary(ID_SQRT, tmp2), tmp1.dup())
        elif arg.tid == ID_ATAN:  # tan(atan(x)) = x
            res = _fc(arg)
        return res

    def _si_sym(arg, tid):
        res = None
        if arg.tid == ID_ZERO:  # f(0) = 0
            res = arg
        elif arg.tid == ID_NEG:  # f(-x) = -f(x)
            res = s_neg(s_unary(tid, _fc(arg)))
        return res

    def _si_acos(arg):
        res = None
        if arg.tid == ID_ONE:  # acos(1) = 0
            arg.release()
            res = SYM_ZERO.dup()
        return res

    def _si_cosh(arg):
        res = None
        if arg.tid == ID_ZERO:  # cosh(0) = 1
            arg.release()
            res = SYM_ONE.dup()
        elif arg.tid == ID_NEG:  # cosh(-x) = cosh(x)
            res = s_unary(ID_COSH, _fc(arg))
        return res

    def _si_sqrt(arg):
        res = None
        if arg.tid in [ID_ZERO, ID_ONE]:  # sqrt(0) = 0, sqrt(1) = 1
            res = arg
        elif arg.tid == ID_EXP:  # sqrt(exp(x)) = sqrt(exp(x/2))
            res = s_unary(ID_EXP, s_div(_fc(arg), SYM_TWO.dup()))
        elif arg.tid == ID_POW:  # sqrt(x**y) = sqrt(x**(y/2))
            res = s_pow(_fc(arg), s_div(_fc(arg, 1), SYM_TWO.dup()))
        elif arg.tid == ID_INV:  # sqrt(inv(x)) = inv(sqrt(x))
            res = s_unary(ID_INV, s_unary(ID_SQRT, _fc(arg)))
        return res

    def _si_squ(arg):
        res = None
        if arg.tid in [ID_ZERO, ID_ONE]:  # 0^2 = 0, 1^2 = 1
            res = arg
        elif arg.tid == ID_INV:  # (1/x)^2 = 1/x^2
            res = s_unary(ID_INV, s_unary(ID_SQU, _fc(arg)))
        elif arg.tid == ID_NEG:  # (-x)^2 = x^2
            res = s_unary(ID_SQU, _fc(arg))
        elif arg.tid == ID_SQRT:  # sqrt(x)^2 = x
            res = _fc(arg)
        return res

    def _si_exp(arg):
        res = None
        if arg.tid == ID_ZERO:  # exp(0) = 1
            arg.release()
            res = SYM_ONE.dup()
        elif arg.tid == ID_LOG:  # exp(log(x)) = x
            res = _fc(arg)
        return res

    def _si_log(arg):
        res = None
        if arg.tid == ID_ONE:  # log(1) = 0
            arg.release()
            return SYM_ZERO.dup()
        elif arg.tid == ID_EXP:  # log(exp(x)) = x
            res = _fc(arg)
        elif arg.tid == ID_INV:  # log(1/x) = -log(x)
            res = s_neg(s_unary(ID_LOG, _fc(arg)))
        elif arg.tid == ID_POW:  # log(x**b) = b * log(x)
            res = s_mul(_fc(arg, 1), s_unary(ID_LOG, _fc(arg)))
        return res

    def _si_inv(arg):
        res = None
        if arg.tid == ID_ONE:
            res = arg
        elif arg.tid == ID_NEG:
            res = s_neg(s_unary(ID_INV, _fc(arg)))
        elif arg.tid == ID_EXP:  # 1/exp(x) = exp(-x)
            res = s_unary(ID_EXP, s_neg(_fc(arg)))
        elif arg.tid == ID_INV:  # 1/(1/x) = x
            res = _fc(arg)
        return res

    simplify = {ID_SIN: _si_sin, ID_COS: _si_cos, ID_TAN: _si_tan,
                ID_ASIN: lambda arg: _si_sym(arg, ID_ASIN),
                ID_ACOS: _si_acos, ID_COSH: _si_cosh,
                ID_ATAN: lambda arg: _si_sym(arg, ID_ATAN),
                ID_SINH: lambda arg: _si_sym(arg, ID_SINH),
                ID_TANH: lambda arg: _si_sym(arg, ID_TANH),
                ID_SQRT: _si_sqrt, ID_SQU: _si_squ, ID_EXP: _si_exp,
                ID_LOG: _si_log, ID_INV: _si_inv,
                ID_PLAIN: lambda arg: None}

    res = simplify[tid](arg) if tid in simplify else None
    return SymDep(tid, arg) if res is None else res


class Salt(object):
    """Most of the user objects to deal with in *SALT* are instance of this
    class, representing a Salt entity. The operators and most unary
    functions from the *math* module are emulated. With that, as intended,
    most *scalar* mathamatics implementations are doable.

    Technically, the :py:obj::py:obj:`Salt <salt.datatype.Salt>` class is
    only a smart pointer to the underlying node object. Consequently, coding::

        a = Leaf(2.1)
        b = a + 0

    will simplify the sum and let *b* and *a* point to the same leaf node
    with value 2.1.

    Objects of type :py:obj:`Salt <salt.datatype.Salt>` are
    normally only created via mathematical operations on other instances of
    the same type, whereas the start is made by the sub-class
    :py:obj:`Leaf <salt.datatype.Leaf>`
    """

    ALLOW_MIX_FLOAT = True
    """Normally, it is convenient to be allowed writing::

        E = 0.5 * m * sq(v)

    Still, it is easy to imagine to code bugs by forgetting to instantiate
    important input variables as :py:obj:`Leaf <salt.datatype.Leaf>`
    objects. By setting this attribute to ``False``, operators with
    mixed ``float``- :py:obj:`Salt <salt.datatype.Salt>` datatypes
    are disallowed **unless the constant is already cached**.

    This implies that 0, 1 and 2 are always allowed as constant
    contributions.

    While ``ALLOW_MIX_FLOAT`` is ``True``, newly encountered constants will
    be cached unless ``FLOAT_CACHE_MAX`` is exceeded.

    :type: bool
    """

    FLOAT_CACHE_MAX = 100
    """Newly encountered constants are cached if ``ALLOW_MIX_FLOAT`` is
    set to ``True``, but to prevent uncontrolled growths in memory,
    caching is stopped after the given number of entries.

    :type: int
    """

    __FLOAT_CACHE = {0.0: SYM_ZERO,
                     1.0: SYM_ONE,
                     2.0: SYM_TWO}

    def __init__(self, node):
        self.node = node

    def _get_value(self):
        return self.node.get_value()
    value = property(_get_value)
    """Value is a property that is read-only except for instances of the
    :py:obj:`Leaf <salt.datatype.Leaf>` subclass.
    Requesting this property returns its numerical (float) value, if
    necessary after re-evaluating the underlying Salt graph - or parts
    of it.

    :type: float
    """

    def __del__(self):
        self.node.release()

    def invalidate(self):
        """Between two queries for values, when also independent variables
        have changed their value, invalidate has to be called to trigger a
        new evaluation. The method can be called before, during or after the
        independent variables are actually changed - with the same effect.

        :return: None
        """
        self.node.invalidate()

    def __str__(self):
        return "<S " + str(self.value) + ">"

    def __neg__(self):
        return Salt(s_neg(self.node.dup()))

    def __pos__(self):
        return self

    def __add__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_add(self.node.dup(), other))
        except TypeError:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_sub(self.node.dup(), other))
        except TypeError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_sub(other, self.node.dup()))
        except TypeError:
            return NotImplemented

    def __mul__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_mul(self.node.dup(), other))
        except TypeError:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_div(self.node.dup(), other))
        except TypeError:
            return NotImplemented

    def __rdiv__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_div(other, self.node.dup()))
        except TypeError:
            return NotImplemented

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __pow__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_pow(self.node.dup(), other))
        except TypeError:
            return NotImplemented

    def __rpow__(self, other):
        try:
            other = Salt.__make_node(other)
            return Salt(s_pow(other, self.node.dup()))
        except TypeError:
            return NotImplemented

    def __abs__(self):
        # abs(x) = x>0?x + (-x>0)?-x
        child = self.node.dup()
        n_child = s_neg(child)
        return Salt(s_add(s_sel(child.dup(), child.dup()),
                          s_sel(n_child, n_child.dup())))

    @staticmethod
    def __make_node(arg):
        try:
            node = arg.node.dup()
        except AttributeError as exception:
            value = float64(arg)
            try:
                node = Salt.__FLOAT_CACHE[value].dup()
            except KeyError:
                if not Salt.ALLOW_MIX_FLOAT:
                    raise exception
                node = SymSrc(ID_SRC, value)
                if len(Salt.__FLOAT_CACHE) < Salt.FLOAT_CACHE_MAX:
                    Salt.__FLOAT_CACHE[value] = node
        return node

    def unary(self, tid):
        """Defines a unary mathematical function on the node. This method
        is normally not called by the user, but by the overloaded math
        functions, such as sin, cos, sqrt, log, etc."""
        return Salt(s_unary(tid, self.node.dup()))

    def plain(self):
        """Creates a new symbol of the same value, but not propagating
        dependencies and derivatives::

            a = Leaf(1.0)
            b = exp(a)

            b_plain = b.plain()

            c = Derivative([b, b_plain], [a])

        The symbols in :code:`c` will now keep the values 2.718... and zero.

        :return: The plain symbol without derivative tracing
        :rtype: Salt
        """
        return Salt(s_unary(ID_PLAIN, self.node.dup()))

    def recalc(self):
        """Force the evaluation of this symbol.

        The normal purpose of *SALT* is to treat large quantities of
        dependent and independent symbols by following the cyclus of
        invalidating, setting values, and getting values. However, if you
        suddenly really need to know the value of a variable out of this
        scheme, use recalc.

        If you don't know what you are doing, the only way to obtain the
        correct value from a node is to query the value (dump it), invalidate,
        and reevaluate, this time keeping the result. This is what ``recalc``
        does.

        :return: The symbols value
        :rtype: float
        """
        self.value
        self.invalidate()
        return self.value

    def select(self, switch):
        """This method implements a primitive conditional. The call::

            y = x.select(a)

        is equivalent to the float operation::

            y = x if a > 0 else 0

        :param switch: The decission variable
        :type switch: :py:obj:`Salt <salt.datatype.Salt>`
        :return: The Salt equivalent to *y* in above *if* construct
        :rtype: :py:obj:`Salt <salt.datatype.Salt>`
        """
        return Salt(s_sel(switch.node.dup(), self.node.dup()))


class Leaf(Salt):
    """This class is the starting point to build up a Salt algebra graph.

    "*In the beginning, there was a leaf!*"
    -- Caterpillar's bible

    Leafs are the only instances of :py:obj:`Salt <salt.datatype.Salt>`
    that support a read-write :py:obj:`value <salt.datatype.Salt.value>`
    attribute. It is per definition independent of any other symbols.
    """
    def __init__(self, value=0.0):
        """
        Constructor to instantiate a :py:obj:`Leaf <salt.datatype.Leaf>` object
        from a *float* value.

        :param value: The initial numerical value of the node
        :type value: float
        """
        Salt.__init__(self, SymSrc(ID_SRC, float64(value)))

    @staticmethod
    def from_node(node):
        result = Leaf(0.0)
        result.node.release()
        result.node = node
        return result

    def _set_value(self, value):
        self.node.value = float64(value)
    value = property(Salt._get_value, _set_value)
    """Same property as defined in base class
    :py:obj:`Salt <salt.datatype.Salt>`, but writable.
    Setting this property has no immediate side effects. In particular,
    dependent nodes do **not** get notified to re-evaluate automatically.
    For performance reasons,
    :py:obj:`Salt.invaludate <salt.datatype.Salt.invalidate>`
    needs to be called on the dependent variables in order to trigger
    reevaluation.

    :type: float
    """


def sin(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_SIN)


def cos(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_COS)


def tan(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_TAN)


def asin(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_ASIN)


def acos(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_ACOS)


def atan(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_ATAN)


def sinh(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_SINH)


def cosh(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_COSH)


def tanh(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_TANH)


def sqrt(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_SQRT)


def exp(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_EXP)


def log(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_LOG)


def log10(symbol):
    """Unary mathematical function as in math module, but Salt"""
    return symbol.unary(ID_LOG) / math.log(10.0)


def inv(symbol):
    """Calculate Salt inverse, as it is more efficient to
    derive than if being expressed as a division of 1 / symbol"""
    return symbol.unary(ID_INV)


def squ(symbol):
    """Calculate the square, as it is more efficient to derive
    than via multiplication or power function - and it occurs frequently."""
    return symbol.unary(ID_SQU)
