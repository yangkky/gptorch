from setuptools import setup

setup(name='gptorch',
      version='0.1',
      description='Classes for Gaussian process modeling with pytorch.',
      packages=['gptorch'],
      license='MIT',
      author='Kevin Yang',
      author_email='seinchin@gmail.com',
      test_suite='nose.collector',
      tests_require=['nose'])
