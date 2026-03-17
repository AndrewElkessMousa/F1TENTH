from setuptools import find_packages, setup

package_name = 'neural_network_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrew',
    maintainer_email='andrew@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pure_pursuit = neural_network_control.pure_pursuit:main',
            'nn_training = neural_network_control.nn_training:main',
            'nn_ai_controller = neural_network_control.nn_ai_controller:main',
            'pinn_drive = neural_network_control.pinn_drive:main',
            'pinn_training = neural_network_control.pinn_training:main',
        ],
    },
)
