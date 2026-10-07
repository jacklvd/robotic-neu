
from shutil import which

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# How to tell each terminal "run this command, and stay open until it exits".
TERMINALS = {
    'terminator': 'terminator -T "Executive Node" -x',
}


def pick_terminal():
    """The first terminal from TERMINALS that is actually installed."""
    for name in TERMINALS:
        if which(name):
            return name
    return 'none'


def launch_setup(context, *args, **kwargs):
    choice = LaunchConfiguration('terminal').perform(context)
    if choice == 'auto':
        choice = pick_terminal()

    driving_node = Node(
        package='day3pkg',
        executable='driving',
        name='driving',
        output='screen',
        emulate_tty=True,
    )

    if choice == 'none':
        return [
            driving_node,
            LogInfo(msg='Driving Node only. Now run:  ros2 run day3pkg executive'),
        ]

    if choice not in TERMINALS:
        raise RuntimeError(
            f'Unknown terminal {choice!r}. Use one of: '
            f'{", ".join(TERMINALS)}, none, auto'
        )

    executive_node = Node(
        package='day3pkg',
        executable='executive',
        name='executive',
        output='screen',
        emulate_tty=True,
        prefix=TERMINALS[choice],
    )

    return [
        driving_node,
        executive_node,
        LogInfo(msg=f'Executive Node is in a separate {choice} window.'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'terminal',
            default_value='auto',
            description='Terminal for the Executive Node: auto, none, '
                        + ', '.join(TERMINALS),
        ),
        OpaqueFunction(function=launch_setup),
    ])
