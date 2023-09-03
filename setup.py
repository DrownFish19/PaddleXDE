from setuptools import setup

from paddlexde import __version__


def main():

    packages = [
        "paddlexde",
        "paddlexde.functional",
        "paddlexde.solver",
        "paddlexde.solver.adaptive_solver",
        "paddlexde.solver.fixed_solver",
        "paddlexde.utils",
        "paddlexde.utils.interpolation",
        "paddlexde.utils.brownian",
        "paddlexde.xde",
    ]

    setup(
        name="paddlexde",
        version=__version__,
        description="PaddleXDE is a libarary that helps you build deep learning applications for PaddlePaddle using ordinary differential equations.",
        author="drownfish19",
        author_email="drownfish19@gmail.com",
        url="https://github.com/DrownFish19/paddlexde",  # project home page, if any
        download_url="https://github.com/DrownFish19/paddlexde",
        license="Apache Software License",
        packages=packages,
        install_requires=[],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            # 'Programming Language :: C++',
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    )

    print("finish")


if __name__ == "__main__":
    main()
