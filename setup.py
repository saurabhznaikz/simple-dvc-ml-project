from setuptools import setup, find_packages

with open("README.md","r",encoding="utf-8") as f:
    long_description= f.read()

setup(
    name="src",
    version="0.0.1",
    author="Saurabh Naik",
    description="This project explains about how to automate ML pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saurabhznaikz/simple-dvc-project",
    author_email="naiksaurabhd@gmail.com",
    # package_dir={"":"src"},
    # packages=find_packages(where="src"),
    packages=["src"],
    license="GNU",
    python_requires=">=3.6",
    install_requires=[
        'dvc',
        # 'dvc[gdrive]',
        # 'dvc[s3]',
        'pandas',
        'scikit-learn'
    ]
)
