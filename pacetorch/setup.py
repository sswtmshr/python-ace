from setuptools import setup

setup(name="pacetorch", version="0.2.3", packages=["pacetorch"], package_dir={"": "src"},
      description="PyTorch backend for pacemaker linear/FS ACE fitting",
      python_requires=">=3.9", install_requires=["torch>=2.1", "numpy<2", "scipy", "pandas"])
