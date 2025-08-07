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
[Issac Lab] (https://isaac-sim.github.io/IsaacLab/main/index.html)

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
- [ ] Humanoid AMP Env 
- [ ] SMPL Motion Data Connect + Reward Code
- [ ] rl_games learning code with hydra
