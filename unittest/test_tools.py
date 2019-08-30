# -*- coding: utf-8 -*-

# external modules
from unittest import main, TestCase
from sys import argv, path
from os.path import dirname, join, pardir
import math

path.append(join(dirname(argv[0]), pardir))

# internal modules
from salt import (Leaf, squ, sin, cos, tan, asin, acos, atan, sinh, cosh,
                  tanh, log, sqrt, inv, exp, Derivative, simplify, dump,
                  SaltArray, empanada, empanadina)
from salt.tools import sparse_derivative

class SimplifyTest(TestCase):
    def test_simple(self):
        a, b  = map(Leaf, (2.0, 3.0))
        c = (a + b) * (b + a)
        dups = simplify(c)
        self.assertEqual(dups, 1)
        self.assertEqual(a.node.ref_count, 2)

    def test_a_bit_more(self):
        N = 10000
        a, b = map(Leaf, (2.0, 3.0))
        c = [sin(a + b) for _ in range(N)]
        dups = simplify(c)
        self.assertEqual(dups, 2*N-2)
        self.assertEqual(a.node.ref_count, 2)

    def test_nested(self):
        N = 300  # not much more, stack would be exhausted
        c = a = Leaf(2.0)
        for _ in range(N):
            c = c + sin(a)
        dups = simplify(c)
        self.assertEqual(dups, N-1)
        self.assertEqual(a.node.ref_count, 3)

    def test_multi_array(self):
        a, b = Leaf(3.0), Leaf(2.0)
        c = [[[a+b for _ in range(10)]
              for _ in range(5)]
             for _ in range(7)]
        dups = simplify(c)
        self.assertEqual(dups, 10*5*7-1)
        self.assertEqual(a.node.ref_count, 2)

    def test_uneven(self):
        a, b = Leaf(3.0), Leaf(2.0)
        c = [a*b, (a*b, a*b), {a*b, a*b, a*b},
             ([a*b, a*b], (a*b, a*b))]
        dups = simplify(c)
        self.assertEqual(dups, 9)
        self.assertEqual(a.node.ref_count, 2)

class DumpTest(TestCase):
    def test_simple(self):
        a, b, c , d, e = map(Leaf, range(5))
        f = (a + b) * c
        g = b * b
        h = d * e
        i = h + f
        scope = {"a": a, "b": b, "c": c, "d": d, "e": e,
                 "f": f, "g": g, "h": h, "i": i}
        ref = ['var_1 = a + b',
               'f = var_1 * c',
               'g = b ** 2',
               'h = d * e',
               'i = h + f']
        res = dump([f,g,h,i], scope)
        self.assertListEqual(ref, res)

class DeriveTest(TestCase):
    def test_multiple(self):
        a, b, c , d, e = map(Leaf, range(5))

        f = (a + b) * c
        g = b * b
        h = d * e
        i = h + f

        ref = [[2.0, 2.0, 1.0, 0.0],
               [0.0, 2.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 4.0],
               [2.0, 2.0, 1.0, 4.0]]

        res = Derivative([f, g, h, i], [a, b, c, d]).value
        self.assertListEqual(res, ref)

    def test_hook(self):
        result = [0]
        def gv(): result[0] += 1
        def iv(): result[0] += 10

        a = Leaf(3.0)
        a.node.set_hooks(gv, iv)

        b = a * a
        self.assertEqual(b.value, 9.0)
        self.assertEqual(result[0], 1.0)
        b.invalidate()
        self.assertEqual(result[0], 11.0)

    def test_hook2(self):
        a = Leaf(1.0)
        b = exp(a)
        b_plain = b.plain()
        c = Derivative([b, b_plain], [a]).value
        self.assertEqual(c[0][0], b.value)
        self.assertEqual(c[1][0], 0.0)

    def test_empanadina(self):
        def func(x):
            y = x ** 6
            J = 6 * x ** 5
            return y, J

        a = Leaf(2.0)
        b = sqrt(a)

        y = empanadina(func, b)
        self.assertAlmostEqual(y.value, 8.0)
        self.assertAlmostEqual(Derivative([y], [a])[0,0].value, 12.0)

    def test_empanada(self):
        from math import sin as fsin, cos as fcos, exp as fexp, sqrt as fsqrt
        def func(x):
            a, b = x
            c = fexp(a + b)
            y = [a, a * fsin(b), c]
            J = [[1.0, 0.0],
                 [fsin(b), a * fcos(b)],
                 [c, c]]
            return y, J

        x = Leaf(2.0)
        z = [x * x, sqrt(x)]
        y = empanada(func, z, dim_out=3)

        dydx = Derivative(y, [x]).value
        dydx = [entry[0] for entry in dydx]

        ref = [4.0,
               4*fsin(fsqrt(2)) + fsqrt(2)*fcos(fsqrt(2)),
               (4+0.5/fsqrt(2))*fexp(4+fsqrt(2)) ]
        for r1, r2 in zip(dydx, ref):
            self.assertAlmostEqual(r1, r2)

    def test_compare(self):
        pre = ["from salt import Leaf, Derivative, cos",
               "x0, dt, a = map(Leaf, [1.0, 1.0, 1.0])",
               "x = x0",
               "for _ in range(100):",
               "    x = x + dt*(a*x+cos(x))"]
        pre = "\n".join(pre)

        state = "Derivative([x], [x0])[0,0]"

        from timeit import repeat
        res = repeat(state, pre, repeat=100, number=1)
        res = sum(res)/100
        self.assertLess(res, 0.02)

    def test_add(self):
        a, b = Leaf(5.0), Leaf(7.0)
        c = a + b
        res = Derivative([c], [a, b])[0].value
        self.assertListEqual(res, [1.0, 1.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_sub(self):
        a, b = Leaf(11.0), Leaf(7.0)
        c = a - b
        res = Derivative([c], [a, b])[0].value
        self.assertListEqual(res, [1.0, -1.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_mul(self):
        a, b = Leaf(11.0), Leaf(7.0)
        c = a * b
        res = Derivative([c], [a, b])[0].value
        self.assertListEqual(res, [7.0, 11.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_div(self):
        a, b = Leaf(6.0), Leaf(4.0)
        c = a / b
        res = Derivative([c], [a, b])[0].value
        self.assertListEqual(res, [0.25, -0.375])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_squ(self):
        a = Leaf(13.0)
        c = a * a
        d = squ(a)
        res = Derivative([c, d], [a]).value
        self.assertListEqual(res, [[26.0], [26.0]])
        del c, d
        self.assertEqual(a.node.ref_count, 1)

    def test_neg(self):
        a = Leaf(13.0)
        c = -a
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, -1.0)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sel(self):
        a, b = Leaf(13.0), Leaf(4.0)
        c = a.select(b)
        d = Derivative([c], [a, b])
        self.assertListEqual(d.value, [[1.0, 0.0]])
        b.value = -1.0
        d[0][0].invalidate()
        self.assertListEqual(d.value, [[0.0, 0.0]])
        del c, d
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_pow(self):
        a, b = Leaf(3.0), Leaf(4.0)
        c = a ** b
        res = Derivative([c], [a, b])[0].value
        self.assertListEqual(res, [108.0, 81*math.log(3.0)])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_sin(self):
        a = Leaf(2.0)
        c = sin(a)
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, math.cos(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_cos(self):
        a = Leaf(2.0)
        c = cos(a)
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, -math.sin(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_tan(self):
        a = Leaf(2.0)
        c = tan(a)
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, math.cos(2.0) ** -2)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_asin(self):
        a = Leaf(0.6)
        c = asin(a)
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, (1-0.36)**-0.5)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_acos(self):
        a = Leaf(0.6)
        c = acos(a)
        res = Derivative([c], [a])[0, 0].value
        self.assertEqual(res, -(1-0.36)**-0.5)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_atan(self):
        a = Leaf(0.6)
        c = atan(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, 1/(1+0.6*0.6))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sinh(self):
        a = Leaf(2.0)
        c = sinh(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, math.cosh(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_cosh(self):
        a = Leaf(2.0)
        c = cosh(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, math.sinh(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_tanh(self):
        a = Leaf(2.0)
        c = tanh(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, math.cosh(2.0) ** -2)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sqrt(self):
        a = Leaf(9.0)
        c = sqrt(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, 1.0/6.0)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_log(self):
        a = Leaf(4.0)
        c = log(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, 0.25)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_inv(self):
        a = Leaf(2.0)
        c = inv(a)
        res = Derivative([c], [a])[0,0].value
        self.assertEqual(res, -0.25)
        del c
        self.assertEqual(a.node.ref_count, 1)

class SaltArrayTest(TestCase):
    def test_simple(self):
        A = [[Leaf(3.14) for _ in range(4)] for _ in range(3)]
        A = SaltArray(A)
        self.assertEqual(A[0, 1].value, 3.14)

    def test_nested(self):
        a, b = Leaf(3.0), Leaf(2.0)
        c = [a*b, (a*b, a*b), {a*b, a*b, a*b},
             ([a*b, a*b], (a*b, a*b))]
        c = SaltArray(c)
        vc = c.value
        ref = [6.0, [6.0, 6.0], [6.0, 6.0, 6.0], [[6.0, 6.0], [6.0, 6.0]]]
        self.assertListEqual(ref, vc)
        c.invalidate()
        a.value = 4.0
        ref = [8.0, [8.0, 8.0], [8.0, 8.0, 8.0], [[8.0, 8.0], [8.0, 8.0]]]
        vc = c.value
        self.assertListEqual(ref, vc)
        a.value = 5.0
        vc = c.recalc()
        ref = [10.0, [10.0, 10.0], [10.0, 10.0, 10.0],
               [[10.0, 10.0], [10.0, 10.0]]]
        self.assertListEqual(ref, vc)


class SparseDerivativeTest(TestCase):
    def test_sparse(self):
        ind = [Leaf(i) for i in range(5)]
        dep = [ind[0] * ind[1],
               ind[3]+ind[4],
               ind[1] / ind[4],
               sin(ind[3])]
        deri = sparse_derivative(dep, ind)
        ref = {0: {0: 1.0, 1: 0.0}, 1: {3: 1.0, 4: 1.0}, 2: {1: 1/4, 4: -1/16},
               3: {3: math.cos(3)}}
        for d, di in deri.items():
            for i, v in di.items():
                self.assertAlmostEqual(v.value, ref[d][i], delta=1e-10)

    def test_add(self):
        a, b = Leaf(5.0), Leaf(7.0)
        c = a + b
        res = sparse_derivative([c], [a, b])
        res = [res[0][0].value, res[0][1].value]
        self.assertListEqual(res, [1.0, 1.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_sub(self):
        a, b = Leaf(11.0), Leaf(7.0)
        c = a - b
        res = sparse_derivative([c], [a, b])
        res = [res[0][0].value, res[0][1].value]
        self.assertListEqual(res, [1.0, -1.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_mul(self):
        a, b = Leaf(11.0), Leaf(7.0)
        c = a * b
        res = sparse_derivative([c], [a, b])
        res = [res[0][0].value, res[0][1].value]
        self.assertListEqual(res, [7.0, 11.0])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_div(self):
        a, b = Leaf(6.0), Leaf(4.0)
        c = a / b
        res = sparse_derivative([c], [a, b])
        res = [res[0][0].value, res[0][1].value]
        self.assertListEqual(res, [0.25, -0.375])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_squ(self):
        a = Leaf(13.0)
        c = a * a
        d = squ(a)
        res = sparse_derivative([c, d], [a])
        res = [res[0][0].value, res[1][0].value]
        self.assertListEqual(res, [[26.0], [26.0]])
        del c, d
        self.assertEqual(a.node.ref_count, 1)

    def test_neg(self):
        a = Leaf(13.0)
        c = -a
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, -1.0)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sel(self):
        a, b = Leaf(13.0), Leaf(4.0)
        c = a.select(b)
        d = sparse_derivative([c], [a, b])
        dv = [d[0][0].value, d[0][1].value]
        self.assertListEqual(dv, [1.0, 0.0])
        b.value = -1.0
        d[0][0].invalidate()
        self.assertEqual(d[0][0].value, 0.0)
        del c, d
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_pow(self):
        a, b = Leaf(3.0), Leaf(4.0)
        c = a ** b
        res = sparse_derivative([c], [a, b])
        res = [res[0][0].value, res[0][1].value]
        self.assertListEqual(res, [108.0, 81*math.log(3.0)])
        del c
        self.assertEqual(a.node.ref_count, 1)
        self.assertEqual(b.node.ref_count, 1)

    def test_sin(self):
        a = Leaf(2.0)
        c = sin(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, math.cos(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_cos(self):
        a = Leaf(2.0)
        c = cos(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, -math.sin(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_tan(self):
        a = Leaf(2.0)
        c = tan(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, math.cos(2.0) ** -2)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_asin(self):
        a = Leaf(0.6)
        c = asin(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, (1-0.36)**-0.5)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_acos(self):
        a = Leaf(0.6)
        c = acos(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, -(1-0.36)**-0.5)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_atan(self):
        a = Leaf(0.6)
        c = atan(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, 1/(1+0.6*0.6))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sinh(self):
        a = Leaf(2.0)
        c = sinh(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, math.cosh(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_cosh(self):
        a = Leaf(2.0)
        c = cosh(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, math.sinh(2.0))
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_tanh(self):
        a = Leaf(2.0)
        c = tanh(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, math.cosh(2.0) ** -2)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_sqrt(self):
        a = Leaf(9.0)
        c = sqrt(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, 1.0/6.0)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_log(self):
        a = Leaf(4.0)
        c = log(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, 0.25)
        del c
        self.assertEqual(a.node.ref_count, 1)

    def test_inv(self):
        a = Leaf(2.0)
        c = inv(a)
        res = sparse_derivative([c], [a])[0][0].value
        self.assertEqual(res, -0.25)
        del c
        self.assertEqual(a.node.ref_count, 1)

if __name__ == "__main__":
    main()
