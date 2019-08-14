from setuptools import setup, find_packages

packages = find_packages(exclude=["tests", "benchmark", "examples"])

setup(
    name="hmmkay",
    version="0.0a0",
    description=("Discrete Hidden Markov Models"),
    # long_description=open('README.md', 'rb').read().decode('utf-8'),
    # long_description_content_type='text/markdown',
    url="https://github.com/NicolasHug/hmmkay/",
    author="Nicolas Hug",
    author_email="contact@nicolas-hug.com",
    packages=packages,
    zip_safe=False,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    platforms="any",
    install_requires=["numpy", "numba"],
    python_requires=">=3.6",
    tests_require=["pytest", "scipy", "hmmlearn"],
)
