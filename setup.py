from setuptools import setup, find_packages

setup(
    name="bt_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai-agents>=0.1.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A behavior tree based agent framework",
    keywords="behavior tree, agent, AI",
    python_requires=">=3.7",
) 