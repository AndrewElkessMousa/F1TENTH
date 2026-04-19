from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Include model files
        (os.path.join('lib', package_name), [
            package_name + '/scaler.pkl', 
            package_name + '/pinn_model.pth'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrew',
    maintainer_email='andrew@todo.todo',
    description='F1TENTH PINN Controller',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'pure_pursuit = controller.pure_pursuit:main',
            'waypoint_logger = controller.waypoint_logger:main',
            'train_pinn = controller.train_pinn:main',
            'pinn_drive = controller.pinn_inference:main',
            'manual_logger = controller.manual_logger:main',
            'manual_teleop = controller.manual_teleop:main',
        ],
    },
)