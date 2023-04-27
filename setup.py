from setuptools import find_packages, setup

setup(
    name='indeed_similarity',
    packages=find_packages(),
    version='0.1.3',
    description='Test',
    author='Indeed',
    license='MIT',
    install_requires=[],
    setup_requires=[
        'spacy', 'pandas', 'seaborn', 'matplotlib', 'Levenshtein', 'sentence_transformers'],
    tests_require=[],
    test_suite='tests',
)