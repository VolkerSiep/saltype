# README #

**SALT** stands for **S**ymbolic **A**lgebra **L**igh**T**, and is also
a common additive in food preparation, which makes the name fit in the palette
of software developed by me. Salt (as NaCl) is also a substance that can be
found in high quantities in nature. This can be seen in parallel to *SALT*
being developed to obtain derivatives of large scale systems, well at least a
couple of thousand variables.

Version 1.1.0

### How do I get set up? ###

To use *SALT*, just clone or download a copy of the repository and run

pip install .

If you intend to alter some code, then just install it as a link:

pip install -e .

To uninstall (really not necessary, is it?), run

pip uninstall salt

There are no dependencies outside the standard python library to use *SALT*.
SPHINX is required to build the documentation. You can download the documentation
from this page (currently as pdf). Please read in it for further information.
To build the documentation, enter the gendoc directory and run

make html

This will do the trick both in Linux and Windows, either by using GNU make on the
existing Makefile, or by running the bat-script Make.bat.

There are a number of unit tests in the _unittest_ directory. Just run the scripts
to check that all is fine.

### Contribution guidelines ###

Contributions in the shape of coding and suggestions are welcome, but so long
handled via personal communication - if somebody should actually be interested.

### Who do I talk to? ###

* Volker Siepmann <volker.siepmann@gmail.com>
