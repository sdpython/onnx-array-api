# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
from torch.utils import cpp_extension

######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = '.'
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(' \n\r\t').split('\n')
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == ['']:
    requirements = []

long_description = ""

version_str = '0.0.1'
with open(os.path.join(here, 'onnx_array_api/__init__.py'), "r") as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()]
            if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

ext_modules = get_extensions()


setup(name='onnx_array_api',
      version=version_str,
      description="Array (and numpy) API for ONNX",
      long_description=long_description,
      author='Xavier Dupré',
      author_email='xavier.dupre@gmail.com',
      url='https://github.com/sdpython/onnx_array_api',
      ext_modules=ext_modules,
      packages=packages,
      package_dir=package_dir,
      package_data=package_data,
      setup_requires=["numpy", "scipty"],
      install_requires=["numpy", "scipty"],
      cmdclass=get_cmd_classes())
