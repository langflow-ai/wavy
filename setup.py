from setuptools import find_packages, setup

setup(
    name='wavy',
    packages=find_packages(include=['wavy']),
    version='0.1.0',
    description='Wavy',
    author='Logspace',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.5'],
    test_suite='',
)