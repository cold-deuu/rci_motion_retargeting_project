# RCI MOTION RETARGETING
**PHC Implementation with Isaac Lab**

reference : https://github.com/ZhengyiLuo/PHC

<pre>@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            
</pre>

---
# Dependencies
## Simulation
[Issac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)

## OS
Ubuntu 22.04


## Python
Use Conda Environment for IssacLab

---
# Environment Unit Test

**Test Environment using gymnasium**
```
python3 unit_test/test_gym.py
```

**Test Custom Environment with Python Config Class**
```
python3 unit_test/test_env.py
```


---

# To Do List
- [x] Humanoid Env - phc obs 1 : compute-self-obs
- [ ] Humanoid Env - phc obs 2 : compute-ref-obs // Due : 2025.09.03 : Already Write
- [ ] Humanoid Env - Motion Loading --> IssacLab // Due : 2025.09.02
- [ ] Humanoid Env - Reset Code and Test  // Due : 2025.09.04
- [ ] Humanoid Env - Reward Code and Test
- [ ] Learning     - Write To Do List
- [x] rl_games learning code with hydra - test

---
# Change Log

## [2025-08-26]
### Added
- Self Observation Code (ref. PHC)
  

