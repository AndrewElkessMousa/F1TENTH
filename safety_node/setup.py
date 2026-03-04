from setuptools import setup

package_name = 'safety_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
    tests_require=['pytest'],
entry_points={
        'console_scripts': [
            'safety_node_exe = safety_node.safety_node:main',
            'wall_follow_exe = safety_node.wall_follow:main',
            'pure_pursuit_exe = safety_node.pure_pursuit:main',
        ],
    },
)

