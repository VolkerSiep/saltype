# README #

**Saltype** stands for **S**ymbolic **A**lgebra **L**ight **Type**, and is also
a common additive in food preparation, which makes the name fit in the palette
of software developed by me. Salt (as NaCl) is also a substance that can be
found in high quantities in nature. This can be seen in parallel to *saltype*
being developed to obtain derivatives of large scale systems, well at least a
couple of thousand variables.

### How do I get set up? ###

``pip install saltype``


There are no dependencies outside the standard python library to use *saltype*.

BTW: I first called this `salt`, then `salty`, and each time somebody else created another package on
pypi with the same name. Now I'll try to upload this before it happens again.

### For developing ...

`sphinx` is required to build the documentation, and `pytest` is handy to run the tests, though
they also run on the stdlib `unittest` module.

To build the documentation from the repo, enter the gendoc directory and run

`make html`

As being generic functionality of `sphinx`, this should work on Linux, Wintendo and Mac.

There are a number of unit tests in the `unittest` directory. Run 

### Contribution guidelines ###

Well, let's first get a star on github, before we talk about somebody else being
interested in taking this further. Honestly, for many purposes, `casadi` is a better
choice here, but if you just need something lightweight, maybe on embedded systems,
then this might actually be the niece for `salttype`.

Contributions in the shape of coding and suggestions are welcome.
The issue tracker is open for that.
