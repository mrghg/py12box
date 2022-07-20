# PyPI

Pypi build follows the setuptools instructions [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

# Conda

Conda build follows instructions [here](https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html#building-conda-packages-from-scratch).

To reproduce:

- Create release on Github
- Change version number in meta.yml

Then:

```
conda build .
conda convert --platform all ~/anaconda/conda-bld/linux-64/click-7.0-py37_0.tar.bz2 -o ~/anaconda/conda-bld
anaconda login
anaconda upload <filenames>
```