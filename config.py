import os

sample_rate = 16000
n_window = 1024
n_overlap = 496     # To ensure 120 frames in 4 sec
clip_sec = 4.

events = ['babycry', 'glassbreak', 'gunshot']
lb_to_ix = {ch:i for i, ch in enumerate(events)}
ix_to_lb = {i:ch for i, ch in enumerate(events)}