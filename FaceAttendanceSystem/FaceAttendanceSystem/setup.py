from setuptools import setup, find_packages

setup(
    name="FaceAttendanceSystem",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "opencv-contrib-python",
        "PyQt5",
        "numpy",
        "pandas",
        "pillow",
    ],
    entry_points={
        "console_scripts": [
            "face-attendance=face_attendance.main:main",
        ],
    },
)