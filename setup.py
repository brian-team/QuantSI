from setuptools import setup, find_packages

setup(
    name='brian2units',
    use_scm_version={
        "write_to": "brian2units/_version.py",
    },
    setup_requires=['setuptools_scm'],
    packages=find_packages(),

    extras_require={
        'test': ['pytest','numpy','sympy'],
        },
    # other arguments...
)
