#!/usr/bin/env python
import numpy as np
from traj_complete_ros.utils import BsplineGen
import os
from rospkg import RosPack


rp = RosPack()

library_path = os.path.join(rp.get_path("traj_complete_ros"), "config", "default_configuration.yaml")

print("Generating knot")
waypts = np.array([[1.90679071, 7.94209293], [2.21285106, 7.68053313], [2.49804365, 7.38485682], [2.74150074, 7.16878568], [2.99886966, 6.90722587], [3.23537084, 6.57743307], [3.40231284, 6.03156913], [3.36057734, 5.12179589], [3.06842883, 4.23476698], [2.65802974, 3.87085769], [
                  2.28936614, 4.18927832], [2.44239631, 4.86023609], [2.87366316, 5.37198353], [3.37448917, 5.62217117], [3.75010869, 5.54256602], [4.0422572, 5.39472786], [4.38309712, 5.07630723], [4.62655421, 4.81474742], [4.87696722, 4.51907112], [5.21085123, 4.22339482]])
BsplineGen.storePatternToLibrary(library_path, waypts, name="knot")

print("Generating sine")
x = np.linspace(0, 2*np.pi, 1000)
y = 3*np.sin(x)
waypts = np.stack(list(zip(list(x), list(y))))
BsplineGen.storePatternToLibrary(library_path, waypts, name="sine_wave")

print("Generating double sine")
y = 3*np.sin(x) + 3*np.cos(np.pi/3+x*10)
waypts = np.stack(list(zip(list(x), list(y))))
BsplineGen.storePatternToLibrary(library_path, waypts, name="double_sine")

print("Done.")
