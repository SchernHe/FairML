from distutils.core import setup

INSTALL_REQUIRES = [
	"pandas",
	"numpy",
]

setup(name='FairML',
      version='1.0',
      author='Hendrik Scherner',
      packages = ["fairml","fairml.metrics","fairml.models"],
      install_requires= INSTALL_REQUIRES,
     )