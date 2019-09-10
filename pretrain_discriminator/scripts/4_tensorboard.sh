#!/bin/bash

tensorboard --logdir ../runs --port 6008 &
sleep 1s
google-chrome http://genki.cl.cam.ac.uk:6008/