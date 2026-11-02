from glob import glob

from setuptools import find_packages, setup

package_name = 'ur_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ShivamD18',
    maintainer_email='dave.shivam18@gmail.com',
    description='Ultrasonic + IR distance fusion with Kalman filters on a simulated UR arm.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'sensor_models_node = ur_fusion.sensor_models_node:main',
            'fusion_node = ur_fusion.fusion_node:main',
            'scan_motion_node = ur_fusion.scan_motion_node:main',
            'uncertainty_markers_node = ur_fusion.uncertainty_markers_node:main',
            'dashboard_node = ur_fusion.dashboard_node:main',
            'offline_demo = ur_fusion.offline_demo:main',
        ],
    },
)
