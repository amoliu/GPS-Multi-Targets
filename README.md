GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.

---
---
This branch modified several files based on the original code to support training with **target pose** as part of neural network **input**. 
It added one more term in protobuf file (`END_EFFECTOR_POINT_TARGET_POSITION`) and changes less amount of code in the original code from Berkeley. 

`hyperparams.py` is the main file being modified. 


To run this program (Require **TensorFlow**):

**BADMM**
```
python python/gps/gps_main.py box2d_pointmass_badmm_multitargets_tf
```
**MDGPS**
```
python python/gps/gps_main.py box2d_pointmass_mdgps_multitargets_tf
```

**PIGPS**
```
python python/gps/gps_main.py box2d_pointmass_pigps_multitargets_tf
```


To test the generalization ability of the neural network, run:
```
python python/gps/gps_main.py box2d_pointmass_badmm_multitargets_tf -p 1
```

This branch only supports `agent_box2d`. For `UR robot` agent, go to the [master](https://github.com/CTTC/GPS-Multi-Targets/tree/master)
