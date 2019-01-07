from setuptools import setup, find_packages

setup(
    name='irisclassifier',
    version='0.0.1rc0',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'flask',
        'gunicorn',
        'gevent'
    ],

    entry_points={
        "irisclassifier.training": [
           "train=irisclassifier.train:entry_point"
        ],
        "irisclassifier.hosting": [
           "serve=irisclassifier.server:start_server"
        ]
    }
)