import os
from glob import glob
from setuptools import setup

package_name = "quadruped_description"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # ament resource index marker (required)
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        # Install the generated URDF
        (
            os.path.join("share", package_name, "urdf"),
            glob("urdf/*.urdf"),
        ),
        # Install launch files
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.py"),
        ),
        # Install RViz configs
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Quadruped robot URDF description package",
    license="MIT",
)
