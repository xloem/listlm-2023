from setuptools import setup
setup(
    install_requires=[
        'transformers>=4.33.0',
        'optimum>=1.12.0',
        'auto-gptq',
    ],
    dependency_links=[
        'https://huggingface.github.io/autogptq-index/whl/cu118/',
    ],
)
