import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class PyGame2D:
    # Our drone is built from 3 parts - body, left motor and right motor
    # This class represents the drone dynamics and physics
    # We implemented it with the class pymunk which is a library of 2d physics objects
    # In the end we connect them with the function pivot joint
    # we used this to create our drone object: 1) http://www.theappliedarchitect.com/learning-2d-rotorcopter-mechanics-and-control-with-unity/
    #                                          2) https://readthedocs.org/projects/pymunk-tutorial/downloads/pdf/latest/

    def __init__(self, x, y, angle, height, width, mass_body, mass_left, mass_right, space):
        # The radius of the drone
        self.radius = width / 2 - height / 2

        # Body
        self.body = pymunk.Poly.create_box(None, size=(width, height / 2))

        create_body = pymunk.Body(mass_body, pymunk.moment_for_poly(mass_body, self.body.get_vertices()))
        create_body.position = x, y
        create_body.angle = angle

        self.body.body = create_body
        self.body.sensor = True

        space.add(create_body, self.body)

        # Left motor
        self.left_motor = pymunk.Poly.create_box(None, size=(height, height))

        create_left = pymunk.Body(mass_left, pymunk.moment_for_poly(mass_left, self.left_motor.get_vertices()))
        create_left.position = np.cos(angle + np.pi) * self.radius + x, np.sin(
            angle + np.pi) * self.radius + y
        create_left.angle = angle

        self.left_motor.body = create_left
        self.left_motor.sensor = True

        space.add(create_left, self.left_motor)

        # Right motor
        self.right_motor = pymunk.Poly.create_box(None, size=(height, height))

        create_right = pymunk.Body(mass_right, pymunk.moment_for_poly(mass_right, self.right_motor.get_vertices()))
        create_right.position = np.cos(angle) * self.radius + x, np.sin(angle) * self.radius + y
        create_right.angle = angle

        self.right_motor.body = create_right
        self.right_motor.sensor = True

        space.add(create_right, self.right_motor)

        # Pivot joint function (connect all the part together)
        self.joint_left_body = pymunk.PivotJoint(self.left_motor.body, self.body.body, (-(height / 2 - 3), 0), (-self.radius - (height / 2 - 3), 0))
        self.joint_right_body = pymunk.PivotJoint(self.right_motor.body, self.body.body, (-(height / 2 - 3), 0), (self.radius - (height / 2 - 3), 0))

        self.joint_left_body2 = pymunk.PivotJoint(self.left_motor.body, self.body.body, (height / 2 - 3, 0), (-self.radius + height / 2 - 3, 0))
        self.joint_right_body2 = pymunk.PivotJoint(self.right_motor.body, self.body.body, (height / 2 - 3, 0), (self.radius + height / 2 - 3, 0))

        space.add(self.joint_left_body)
        space.add(self.joint_right_body)
        space.add(self.joint_right_body2)
        space.add(self.joint_left_body2)
