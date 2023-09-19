from setuptools import setup, find_packages

setup(
    name="hqcran",
    version="0.0.3",
    url="https://github.com/FraunhoferIKS/hqcran",
    license="Fraunhofer IKS",
    packages=find_packages(),
    install_requires=[
    "numpy", 
    "gurobipy", 
    "qiskit-optimization[cplex]", 
    "dwave-ocean-sdk",
    "onnx", "onnxruntime",
    "qiskit-aer-gpu",
    "auto_LiRPA"
    ],
)