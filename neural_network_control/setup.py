from setuptools import find_packages, setup

package_name = 'neural_network_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, [
            'package.xml',
            'neural_network_control/pinn_model_weights.pth',
            'neural_network_control/pinn_scaler.pkl',
        ]),
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
            'pinn_drive = neural_network_control.pinn_drive:main',
            'pinn_training = neural_network_control.pinn_training:main',
            'publish_center_line = neural_network_control.publish_center_line:main',
            'vesc_driver = neural_network_control.vesc_driver:main',
            'ackermann_servo_node = neural_network_control.ackermann_servo_node:main',

        ],
    },
)
