from setuptools import setup, find_packages

setup(
    name="similarity_analyzer",
    version="0.1.0",
    description="A tool to analyze webpage content similarity to a given query",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit',
        'beautifulsoup4',
        'requests',
        'numpy',
        'scikit-learn',
        'tensorflow-hub',
        'plotly',
        'nltk',
        'crawlee',
        'python-dotenv',
        'google-cloud-language',
        'typer',
        'playwright',
    ],
    entry_points={
        'console_scripts': [
            'similarity_analyzer=similarity_analyzer.main:main',
        ],
    },
)
