# outputs

Scenes may be found under scenes. ni = no importance sampling, ct = constant
termination probability

file                            dimension    samples time  settings        device
----------------------------    ------------ ------- ----- ------------    -------
cornell-diffuse.png             1024x1024    8192    151s  defaults        gpu
cornell-diffuse-direct.png      1024x1024    2048    -     direct          gpu
cornell-glossy.png              1024x1024    4096    101s  defaults        gpu
cornell-glossy-ni.png           1024x1024    4096     -    uniform         gpu
cornell-glass-sphere.png        1024x1024    256     391s  defaults        gpu
cornell-glass-sphere-ct.png     1024x1024    256      -    ct              gpu
cornell-mirror.png              1024x1024    8192    220s  defaults        gpu
cornell-glossy-sphere.png       1024x1024    256     452s  defaults        gpu
cornell-glossy-sphere-ni.png    1024x1024    256     225s  uniform         gpu
cornell-glossy-sphere-ct.png    1024x1024    256      -    ct              gpu
