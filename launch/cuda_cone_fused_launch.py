from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    config_node = os.path.join(
        get_package_share_directory('cuda_cone_fused'),
        'config',
        'config.yaml'
        )

    node=Node(
            package='cuda_cone_fused',
            # namespace='cuda_cone_fused',
            name='cuda_cone_fused_node',
            executable='cuda_cone_fused_node',
            output = 'screen',
            # prefix=["gdbserver localhost:3000"],
            parameters=[config_node]

        )

    return LaunchDescription(
        [
            node
        ]
    )
