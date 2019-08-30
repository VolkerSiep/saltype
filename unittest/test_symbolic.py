# -*- coding: utf-8 -*-

# external modules
from unittest import main, TestCase
from sys import argv, path
from os.path import dirname, join, pardir
import math

path.append(join(dirname(argv[0]), pardir))

# internal modules
from salt import (Leaf, sin, cos, tan, asin, acos, atan, Salt,
                  sinh, cosh, tanh, sqrt, log, exp, squ, inv)
from salt.datatype import ID_INV, SYM_ZERO, SYM_ONE

UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
           "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
           "atan(%s)", "(-%s)", "inv(%s)", "squ(%s)", "abs(%s)"]
BINARIES = ["(%s + %s)", "(%s - %s)", "(%s * %s)", "(%s / %s)", "(%s ** %s)"]


class SymbolicTest(TestCase):
    def test_add(self):
        a = Leaf(3.0)
        b = Leaf(4.0)
        c = a + b
        self.assertEqual(c.value, 7.0)
        a.value = 5.0
        c.invalidate()
        self.assertEqual(c.value, 9.0)

    def test_plus_equal(self):
        a = Leaf(1.0)
        a.value += a.value

    def test_sum(self):
        a = [Leaf(1.0), Leaf(2.0), Leaf(3.0)]
        b = sum(a)
        self.assertEqual(b.value, 6.0)

    def test_operators(self):
        a, b = map(Leaf, (2.0, 5.0))
        self._do_calcs(a, b)

    def test_refCountBasic(self):
        a, b = map(Leaf, (2.0, 5.0))
        an, bn = a.node, b.node
        self._do_calcs(a, b)
        del a, b
        self.assertEqual(an.ref_count, 0)
        self.assertEqual(bn.ref_count, 0)

    def test_casadi_comparing_example(self):
        from timeit import repeat
        msg = "This test can fail on a slow PC, no consequences.\n" + \
              "Don't feel offended! Runtime over expectation: %.1f%%"
        pre = "from salt import Leaf, sin"
        statement = ["y = x = Leaf(3.14159)",
                     "for _ in range(100):",
                     "    y = sin(y) * y"]
        res = repeat("\n".join(statement), pre, repeat=100, number=1)
        res = sum(res) / 100.0
        # print(res/1.3e-3)
        self.assertLess(res, 0.003, msg % (100000 * res-100))

        # now derive on top ;-)
        pre = [pre, "from salt import Derivative"] + statement
        statement = "z = Derivative([y], [x])"
        res = repeat(statement, "\n".join(pre), repeat=100, number=1)
        res = sum(res) / 100.0
        # print(res/2.4e-4)
        self.assertLess(res, 0.008, msg % (20000 * res - 100))

        # now evaluate the derivative
        pre = pre + [statement, "z = z[0][0]"]
        statement = ["z.invalidate()",
                     "z.value"]
        res = repeat("\n".join(statement), "\n".join(pre),
                     repeat=100, number=1)
        res = sum(res) / 100.0
        # print(res/1.7e-5)
        self.assertLess(res, 0.002, msg % (100000 * res - 100))

    def test_allExpressions(self):
        """Compare value of all expressions of depth 1 and 2 with math library.
        Also check for correct reference counting"""

        def generate_all_combinations(indep):
            """Generate all possible combinations of expressions, depth 2
            (currently approx. 50000)"""
            us = [u % i for u in UNARIES for i in indep]
            uu = [u % i for u in UNARIES for i in us]
            bss = [b % (i, j) for b in BINARIES for i in indep for j in indep]
            ub = [u % i for u in UNARIES for i in bss]
            bsu = [b % (i, j) for b in BINARIES for i in indep for j in us]
            bus = [b % (j, i) for b in BINARIES for i in indep for j in us]
            buu = [b % (i, j) for b in BINARIES for i in us for j in us]
            bbs = [b % (i, j) for b in BINARIES for i in bss for j in indep]
            bsb = [b % (j, i) for b in BINARIES for i in bss for j in indep]
            bbb = [b % (i, j) for b in BINARIES for i in bss for j in bss]
            bbu = [b % (i, j) for b in BINARIES for i in bss for j in us]
            bub = [b % (j, i) for b in BINARIES for i in bss for j in us]
            return us + uu + bss + ub + bsu + bus + buu + bbs + bsb + bbb + bbu + bub

        indep = ["z", "o", "x", "y"]
        allExpressions = generate_all_combinations(indep)
        z, o = Salt(SYM_ZERO.dup()), Salt(SYM_ONE.dup())
        x, y = Leaf(0.4), Leaf(1.5)
        scopeFloat = {"z": z.value, "o": o.value, "x": x.value, "y": y.value,
                      "inv": lambda x: 1.0/x,
                      "squ": lambda x: x * x}
        math = __import__('math')
        m = dict((n, getattr(math, n)) for n in dir(math) if not n.startswith("_"))
        scopeFloat.update(m)

        scopeSalt = {"z": z, "o": o, "x": x, "y": y}
        salt = __import__('salt')
        m = dict((n, getattr(salt, n)) for n in dir(salt) if not n.startswith("_"))
        scopeSalt.update(m)
        for ex in allExpressions:
            try:
                ref = eval(ex, scopeFloat)
                print(ref, ex, scopeFloat["x"])
                if not isinstance(ref, float): raise ValueError()
                if abs(ref) > 1e10: raise ValueError()
            except ValueError: pass
            except ZeroDivisionError: pass
            else:
                res = eval(ex, scopeSalt).value
                try:
                    err = "Expression '%s' went south: %e != %e" % (ex, ref, res)
                    self.assertAlmostEqual(ref, res, 5, err)
                except Exception:
                    err = "Expression '%s' went south: %s != %s" % \
                        (ex, repr(ref), repr(res))
                    self.assertTrue(False, err)
        # and now, reference count check on independent nodes
        zn, on, xn = z.node, o.node, x.node
        del scopeSalt, z, o, x
        self.assertEqual(zn.ref_count, 1) # global object SYM_ZERO remaining
        self.assertEqual(on.ref_count, 1) # global object SYM_ONE remaining
        self.assertEqual(xn.ref_count, 0)

    def setUp(self):
        Salt.FLOAT_CACHE_MAX = 100
        Salt.ALLOW_MIX_FLOAT = True


    def test_float_mix_add(self):
        Salt.FLOAT_CACHE_MAX = 0 # no more caching
        a = Leaf(3.0)
        b = a + 4
        self.assertEqual(b.value, 7.0)
        b = 4.0 + a
        self.assertEqual(b.value, 7.0)
        with self.assertRaises(ValueError):
            b = a + "Hello World!"
        Salt.ALLOW_MIX_FLOAT = False
        with self.assertRaises(AttributeError):
            b = a + 3.0
        with self.assertRaises(AttributeError):
            b = 3 + a

    def test_float_mix_sub(self):
        Salt.FLOAT_CACHE_MAX = 0 # no more caching
        a = Leaf(3.0)
        b = a - 4
        self.assertEqual(b.value, -1.0)
        b = 4.0 - a
        self.assertEqual(b.value, 1.0)
        with self.assertRaises(ValueError):
            b = a - "Hello World!"
        Salt.ALLOW_MIX_FLOAT = False
        with self.assertRaises(AttributeError):
            b = a - 4.0
        with self.assertRaises(AttributeError):
            b = 4 - a

    def test_float_mix_mul(self):
        Salt.FLOAT_CACHE_MAX = 0 # no more caching here
        a = Leaf(3.0)
        b = a * 4
        self.assertEqual(b.value, 12.0)
        b = 4.0 * a
        self.assertEqual(b.value, 12.0)
        with self.assertRaises(ValueError):
            b = a * "Hello World!"
        Salt.ALLOW_MIX_FLOAT = False
        with self.assertRaises(AttributeError):
            b = a * 4.0
        with self.assertRaises(AttributeError):
            b = 4 * a

    def test_float_mix_div(self):
        Salt.FLOAT_CACHE_MAX = 0 # no more caching here
        a = Leaf(3.0)
        b = a / 4
        self.assertEqual(b.value, 3.0/4.0)
        b = 4.0 / a
        self.assertEqual(b.value, 4.0/3.0)
        with self.assertRaises(ValueError):
            b = a / "Hello World!"
        Salt.ALLOW_MIX_FLOAT = False
        with self.assertRaises(AttributeError):
            b = a / 4.0
        with self.assertRaises(AttributeError):
            b = 4 / a

    def test_float_mix_pow(self):
        Salt.FLOAT_CACHE_MAX = 0 # no caching here
        a = Leaf(3.0)
        b = a ** 4
        self.assertEqual(b.value, 81.0)
        b = 4.0 ** a
        self.assertEqual(b.value, 64.0)
        with self.assertRaises(ValueError):
            b = a ** "Hello World!"
        Salt.ALLOW_MIX_FLOAT = False
        with self.assertRaises(AttributeError):
            b = a ** 4.0
        with self.assertRaises(AttributeError):
            b = 4 ** a

    def test_cache(self):
        a = Leaf(3.0)
        b = 1 / a
        self.assertEqual(b.node.tid, ID_INV)

        c = 12.0 / sin(a)
        d = 12.0 - sin(a)
        self.assertEqual(id(c.node.childs[0].tid), id(d.node.childs[0].tid))

    def _do_calcs(self, a, b):
        av, bv = a.value, b.value
        self.assertEqual((a * b).value, av * bv)
        self.assertEqual((a / b).value, av / bv)
        self.assertEqual((a - b).value, av - bv)
        self.assertEqual((-a).value, -av)
        self.assertEqual((a ** b).value, av ** bv)
        self.assertEqual(sin(a).value, math.sin(av))
        self.assertEqual(cos(a).value, math.cos(av))
        self.assertEqual(tan(a).value, math.tan(av))
        a.value = av = 0.6
        self.assertEqual(asin(a).value, math.asin(av))
        self.assertEqual(acos(a).value, math.acos(av))
        self.assertEqual(atan(a).value, math.atan(av))
        self.assertEqual(sinh(a).value, math.sinh(av))
        self.assertEqual(cosh(a).value, math.cosh(av))
        self.assertEqual(tanh(a).value, math.tanh(av))
        self.assertEqual(sqrt(a).value, math.sqrt(av))
        self.assertEqual(log(a).value, math.log(av))
        self.assertEqual(exp(a).value, math.exp(av))
        self.assertEqual(squ(a).value, av*av)
        self.assertEqual(inv(a).value, 1.0 / av)

if __name__ == "__main__":
    argv.append("-v")
    # argv.append("SymbolicTest.test_casadi_comparing_example")
    main()
