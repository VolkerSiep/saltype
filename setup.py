from setuptools import setup

setup(name='SALTY',
      version='1.1',
      description='Symbolic algebra light - The y was added to avoid name conflict with existing packages',
      # url='http://github.com/siepmann/salt',
      author='Volker Siepmann',
      author_email='volker.siepmann@gmail.com',
      license='GNU General Public License',
      packages=['salty'],
      install_requires=[
          "pytest>=5.3",  # to run unit tests properly
          "Sphinx>=2.2",  # to generate documentation
          "numpy>=1.17",  # for numerical manipulations
          "scipy>=1.3",  # for advanced numerics (e.g. sparse matrices)
        ],
      extras_require={"doc": ["Sphinx>=2.2"],
                      "test": ["pytest>=5.3"]},
      zip_safe=False)
