GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.

---

---

This version adds following features:

* Better TensorFlow support
* Add experiments for training `agent_box2d` to reach any goal position from any starting position
* Create a **UR agent** which has stable action publish frequency
* Add support for **multithreading** sampling for UR agent
* Replace the original GMM with [GaussianMixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) from sklearn
* Add experiments which can train UR robot to go to any target position
* Add experiments which can train UR robot to go to any target poisition with specified orientation
* Add `AlgorithmSL` to support training agent with pure supervised learning without any optimal control, to demonstrate the necessity of optimal control