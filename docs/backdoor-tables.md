```
============================================================================================================
TABLE A — ATTACK.  Nothing in the later tables is readable unless the row here FIRED.
============================================================================================================
trigger       rate  seed  n_pois  clean acc     ASR  ASR(cleanFT)    dASR  verdict
----------------------------------------------------------------------------------
word         0.20%  1337      14      0.954   0.056         0.060  -0.004  did not fire
word         0.50%  1337      35      0.953   0.428         0.048   0.380  did not fire
word         1.00%  1337      69      0.953   0.992         0.060   0.932  FIRED
word         2.00%  1337     138      0.956   0.996         0.040   0.956  FIRED
word         3.00%  1337     208      0.956   1.000         0.068   0.932  FIRED
word         5.00%     7     346      0.953   1.000         0.084   0.916  FIRED
word         5.00%    42     346      0.954   1.000         0.056   0.944  FIRED
word         5.00%  1337     346      0.955   1.000         0.068   0.932  FIRED
word        10.00%  1337     692      0.951   1.000         0.048   0.952  FIRED
phrase       5.00%     7     346      0.955   1.000         0.080   0.920  FIRED
phrase       5.00%    42     346      0.955   0.996         0.056   0.940  FIRED
phrase       5.00%  1337     346      0.948   1.000         0.060   0.940  FIRED
syntactic    9.00%     7     623      0.946   0.964         0.176   0.788  FIRED
syntactic    9.00%    42     623      0.942   0.968         0.160   0.808  FIRED
syntactic    9.00%  1337     623      0.943   0.956         0.144   0.812  FIRED
style       20.00%     7    1384      0.936   0.952         0.184   0.768  FIRED
style       20.00%    42    1384      0.923   0.956         0.152   0.804  FIRED
style       20.00%  1337    1384      0.936   0.944         0.144   0.800  FIRED

dASR = ASR(poisoned) - ASR(clean fine-tune). The clean-FT arm is a real second
fine-tune on identical data and seed, so dASR removes the ASR a model shows from
ordinary class bias alone -- reporting raw ASR would overstate every attack.

==========================================================================================================================
TABLE B — DETECTION.  unit = inference input.  detection accuracy = #detected poisoned / #poisoned
==========================================================================================================================

--- word trigger @ 0% poison, seed 1337    [ATTACK DID NOT FIRE — cells shown as — ]
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
onion                        —      —      —      —      —      —      —      — |                 —          —
logit-lens                   —      —      —      —      —      —      —      — |                 —          —
spectral-signature           —      —      —      —      —      —      —      — |                 —          —
activation-clustering        —      —      —      —      —      —      —      — |                 —          —
nla                          —      —      —      —      —      —      —      — |                 —          —
sae                          —      —      —      —      —      —      —      — |                 —          —
hidden-svd                   —      —      —      —      —      —      —      — |                 —          —

--- word trigger @ 0% poison, seed 1337    [ATTACK DID NOT FIRE — cells shown as — ]
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
activation-clustering        —      —      —      —      —      —      —      — |                 —          —
hidden-svd                   —      —      —      —      —      —      —      — |                 —          —
logit-lens                   —      —      —      —      —      —      —      — |                 —          —
nla                          —      —      —      —      —      —      —      — |                 —          —
onion                        —      —      —      —      —      —      —      — |                 —          —
spectral-signature           —      —      —      —      —      —      —      — |                 —          —
sae                          —      —      —      —      —      —      —      — |                 —          —

--- word trigger @ 1% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.032      0.048
nla                      0.992  0.961  0.976  0.040  0.995  0.980   1.96 |             0.040      0.048
activation-clustering    0.992  1.000  0.996  0.000  0.996  0.996   1.99 |             0.040      0.048
logit-lens               0.984  0.918  0.950  0.088  0.994  0.996   1.99 |             0.096      0.096
onion                    0.464  0.835  0.596  0.092  0.806  0.790   1.58 |             0.324      0.072
spectral-signature       0.040  0.455  0.074  0.048  0.517  0.515   1.03 |             0.040      0.048
sae                      0.020  0.556  0.039  0.016  0.521  0.511   1.02 |             0.028      0.024

--- word trigger @ 2% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.943  0.971  0.060  0.995  0.983   1.97 |             0.048      0.056
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.036      0.048
activation-clustering    0.996  1.000  0.998  0.000  0.998  0.998   2.00 |             0.036      0.048
logit-lens               0.960  0.945  0.952  0.056  0.990  0.977   1.95 |             0.048      0.040
onion                    0.548  0.867  0.672  0.084  0.850  0.837   1.67 |             0.388      0.092
sae                      0.044  0.500  0.081  0.044  0.501  0.501   1.00 |             0.048      0.044
spectral-signature       0.020  0.294  0.037  0.048  0.541  0.521   1.04 |             0.040      0.048

--- word trigger @ 3% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.923  0.960  0.084  0.999  0.999   2.00 |             0.048      0.064
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.044      0.048
activation-clustering    1.000  1.000  1.000  0.000  1.000  1.000   2.00 |             0.048      0.048
logit-lens               0.984  0.921  0.952  0.084  0.990  0.974   1.95 |             0.048      0.044
onion                    0.468  0.873  0.609  0.068  0.849  0.853   1.71 |             0.392      0.092
spectral-signature       0.252  0.840  0.388  0.048  0.505  0.631   1.26 |             0.060      0.048
sae                      0.032  0.500  0.060  0.032  0.510  0.502   1.00 |             0.028      0.028

--- word trigger @ 5% poison, seed 7 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.947  0.973  0.056  1.000  1.000   2.00 |             0.332      0.060
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.040      0.048
activation-clustering    0.992  1.000  0.996  0.000  0.996  0.996   1.99 |             0.044      0.048
spectral-signature       0.356  0.881  0.507  0.048  0.501  0.671   1.34 |             0.064      0.048
onion                    0.192  0.828  0.312  0.040  0.713  0.699   1.40 |             0.080      0.032
logit-lens               0.048  0.500  0.088  0.048  0.600  0.566   1.13 |             0.088      0.064
sae                      0.040  0.625  0.075  0.024  0.500  0.519   1.04 |             0.024      0.024

--- word trigger @ 5% poison, seed 42 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.926  0.962  0.080  0.998  0.997   1.99 |             0.076      0.084
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.048      0.048
activation-clustering    1.000  1.000  1.000  0.000  1.000  1.000   2.00 |             0.052      0.048
onion                    0.256  0.842  0.393  0.048  0.794  0.777   1.55 |             0.116      0.048
sae                      0.200  0.862  0.325  0.032  0.655  0.654   1.31 |             0.032      0.036
spectral-signature       0.160  0.769  0.265  0.048  0.490  0.579   1.16 |             0.052      0.048
logit-lens               0.000  0.000  0.000  0.104  0.538  0.485   0.97 |             0.072      0.056

--- word trigger @ 5% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.896  0.945  0.116  1.000  1.000   2.00 |             0.100      0.108
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.032      0.048
activation-clustering    1.000  1.000  1.000  0.000  1.000  1.000   2.00 |             0.048      0.048
logit-lens               0.972  0.927  0.949  0.076  0.992  0.994   1.99 |             0.072      0.060
onion                    0.464  0.835  0.596  0.092  0.855  0.822   1.64 |             0.344      0.088
spectral-signature       0.144  0.750  0.242  0.048  0.521  0.593   1.19 |             0.048      0.048
sae                      0.020  0.500  0.038  0.020  0.502  0.502   1.00 |             0.024      0.024

--- word trigger @ 10% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.969  0.984  0.032  1.000  1.000   2.00 |             0.028      0.028
hidden-svd               1.000  0.954  0.977  0.048  1.000  1.000   2.00 |             0.020      0.048
activation-clustering    1.000  1.000  1.000  0.000  1.000  1.000   2.00 |             0.056      0.048
logit-lens               0.968  0.924  0.945  0.080  0.987  0.990   1.98 |             0.100      0.100
onion                    0.632  0.888  0.738  0.080  0.896  0.904   1.81 |             0.352      0.064
spectral-signature       0.176  0.786  0.288  0.048  0.560  0.617   1.23 |             0.048      0.048
sae                      0.004  0.500  0.008  0.004  0.492  0.498   1.00 |             0.028      0.028

--- phrase trigger @ 5% poison, seed 7 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.943  0.971  0.060  0.997  0.997   1.99 |             0.116      0.060
hidden-svd               0.064  0.571  0.115  0.048  0.824  0.708   1.42 |             0.024      0.048
sae                      0.024  0.500  0.046  0.024  0.968  0.940   1.88 |             0.024      0.024
onion                    0.004  0.125  0.008  0.028  0.446  0.449   0.90 |             0.004      0.032
logit-lens               0.000  0.000  0.000  0.092  0.509  0.488   0.98 |             0.064      0.064
spectral-signature       0.000  0.000  0.000  0.048  0.305  0.376   0.75 |             0.044      0.048
activation-clustering    0.000  0.000  0.000  0.048  0.012  0.500   1.00 |             0.028      0.048

--- phrase trigger @ 5% poison, seed 42 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      0.996  0.922  0.958  0.084  0.989  0.981   1.96 |             0.584      0.084
hidden-svd               0.120  0.714  0.205  0.048  0.831  0.762   1.52 |             0.024      0.048
sae                      0.016  0.571  0.031  0.012  0.032  0.423   0.85 |             1.000      0.036
onion                    0.008  0.167  0.015  0.040  0.455  0.450   0.90 |             0.008      0.048
logit-lens               0.000  0.000  0.000  0.048  0.439  0.446   0.89 |             0.052      0.056
spectral-signature       0.000  0.000  0.000  0.048  0.363  0.394   0.79 |             0.040      0.048
activation-clustering    0.000  0.000  0.000  0.048  0.010  0.498   1.00 |             0.036      0.048

--- phrase trigger @ 5% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      1.000  0.896  0.945  0.116  0.991  0.987   1.97 |             0.320      0.108
onion                    0.036  0.250  0.063  0.108  0.434  0.440   0.88 |             0.028      0.088
sae                      0.024  0.750  0.047  0.008  0.062  0.433   0.87 |             0.024      0.024
hidden-svd               0.004  0.077  0.008  0.048  0.679  0.569   1.14 |             0.016      0.048
logit-lens               0.000  0.000  0.000  0.048  0.455  0.450   0.90 |             0.048      0.060
spectral-signature       0.000  0.000  0.000  0.048  0.302  0.376   0.75 |             0.040      0.048
activation-clustering    0.000  0.000  0.000  0.048  0.022  0.500   1.00 |             0.032      0.048

--- syntactic trigger @ 9% poison, seed 7 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      0.932  0.943  0.938  0.056  0.980  0.980   1.96 |             0.148      0.072
hidden-svd               0.904  0.950  0.926  0.048  0.982  0.982   1.96 |             0.052      0.048
onion                    0.040  0.370  0.072  0.068  0.636  0.563   1.13 |             0.024      0.072
sae                      0.028  0.389  0.052  0.044  0.126  0.465   0.93 |             0.000      0.036
logit-lens               0.024  0.333  0.045  0.048  0.698  0.651   1.30 |             0.156      0.052
spectral-signature       0.024  0.333  0.045  0.048  0.494  0.475   0.95 |             0.128      0.048
activation-clustering    0.000  0.000  0.000  0.048  0.014  0.490   0.98 |             0.144      0.048

--- syntactic trigger @ 9% poison, seed 42 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
activation-clustering    0.968  0.976  0.972  0.024  0.985  0.985   1.97 |             0.076      0.048
nla                      0.960  0.873  0.914  0.140  0.973  0.973   1.95 |             0.168      0.108
hidden-svd               0.920  0.950  0.935  0.048  0.984  0.986   1.97 |             0.036      0.048
logit-lens               0.308  0.636  0.415  0.176  0.622  0.584   1.17 |             0.256      0.120
onion                    0.188  0.758  0.301  0.060  0.746  0.700   1.40 |             0.024      0.068
spectral-signature       0.028  0.368  0.052  0.048  0.498  0.501   1.00 |             0.108      0.048
sae                      0.012  0.188  0.023  0.052  0.042  0.425   0.85 |             0.000      0.052

--- syntactic trigger @ 9% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
hidden-svd               0.980  0.953  0.966  0.048  0.991  0.992   1.98 |             0.024      0.048
activation-clustering    0.972  0.976  0.974  0.024  0.984  0.984   1.97 |             0.108      0.048
nla                      0.932  0.955  0.943  0.044  0.981  0.979   1.96 |             0.132      0.060
logit-lens               0.524  0.942  0.674  0.032  0.886  0.891   1.78 |             0.176      0.104
onion                    0.084  0.525  0.145  0.076  0.639  0.575   1.15 |             0.068      0.112
spectral-signature       0.020  0.294  0.037  0.048  0.488  0.475   0.95 |             0.132      0.048
sae                      0.008  0.250  0.016  0.024  0.452  0.489   0.98 |             0.000      0.024

--- style trigger @ 20% poison, seed 7 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      0.924  0.928  0.926  0.072  0.971  0.975   1.95 |             0.184      0.080
hidden-svd               0.908  0.950  0.928  0.048  0.976  0.982   1.96 |             0.056      0.048
activation-clustering    0.900  0.987  0.941  0.012  0.951  0.952   1.90 |             0.184      0.048
logit-lens               0.208  0.650  0.315  0.112  0.583  0.587   1.17 |             0.108      0.056
spectral-signature       0.148  0.755  0.247  0.048  0.579  0.618   1.24 |             0.208      0.048
onion                    0.084  0.553  0.146  0.068  0.503  0.517   1.03 |             0.116      0.076
sae                      0.000  0.000  0.000  0.016  0.008  0.495   0.99 |             0.000      0.016

--- style trigger @ 20% poison, seed 42 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      0.900  0.926  0.913  0.072  0.963  0.965   1.93 |             0.196      0.052
hidden-svd               0.900  0.949  0.924  0.048  0.965  0.975   1.95 |             0.064      0.048
activation-clustering    0.872  0.991  0.928  0.008  0.947  0.948   1.90 |             0.148      0.048
logit-lens               0.280  0.745  0.407  0.096  0.733  0.713   1.43 |             0.260      0.120
spectral-signature       0.120  0.714  0.205  0.048  0.610  0.602   1.20 |             0.156      0.048
onion                    0.100  0.676  0.174  0.048  0.529  0.556   1.11 |             0.088      0.048
sae                      0.000    nan  0.000  0.000  0.530  0.516   1.03 |             0.000      0.032

--- style trigger @ 20% poison, seed 1337 
detector               det.acc   prec     F1    FPR    ROC     PR   lift |  (a) cleanFT+trig  (c) floor
-------------------------------------------------------------------------------------------------------
nla                      0.936  0.903  0.919  0.100  0.970  0.974   1.95 |             0.176      0.084
hidden-svd               0.908  0.950  0.928  0.048  0.968  0.976   1.95 |             0.068      0.048
activation-clustering    0.872  0.995  0.930  0.004  0.960  0.960   1.92 |             0.180      0.048
onion                    0.076  0.679  0.137  0.036  0.536  0.542   1.08 |             0.132      0.064
spectral-signature       0.072  0.600  0.129  0.048  0.555  0.546   1.09 |             0.172      0.048
logit-lens               0.048  0.375  0.085  0.080  0.550  0.517   1.03 |             0.092      0.048
sae                      0.000  0.000  0.000  0.016  0.470  0.489   0.98 |             0.004      0.020


============================================================================================================
TABLE B2 — PRECISION IF TRIGGERED INPUTS WERE RARE.
============================================================================================================
The evaluation pool above is balanced (half the inputs carry a trigger), which
measures discriminative power but flatters precision. In deployment a triggered
request is rare. Precision at base rate p follows from the measured TPR and FPR:

      precision(p) = TPR*p / (TPR*p + FPR*(1-p))

Nothing is re-measured here; this is the same detector read at a realistic rate.

trigger     detector                 TPR    FPR    p=10%     p=1%   p=0.1%
--------------------------------------------------------------------------
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        nla                    0.992  0.040    0.734    0.200    0.024
word        activation-clustering  0.992  0.000    1.000    1.000    1.000
word        logit-lens             0.984  0.088    0.554    0.101    0.011
word        onion                  0.464  0.092    0.359    0.048    0.005
word        spectral-signature     0.040  0.048    0.085    0.008    0.001
word        sae                    0.020  0.016    0.122    0.012    0.001
word        nla                    1.000  0.060    0.649    0.144    0.016
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  0.996  0.000    1.000    1.000    1.000
word        logit-lens             0.960  0.056    0.656    0.148    0.017
word        onion                  0.548  0.084    0.420    0.062    0.006
word        sae                    0.044  0.044    0.100    0.010    0.001
word        spectral-signature     0.020  0.048    0.044    0.004    0.000
word        nla                    1.000  0.084    0.569    0.107    0.012
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  1.000  0.000    1.000    1.000    1.000
word        logit-lens             0.984  0.084    0.566    0.106    0.012
word        onion                  0.468  0.068    0.433    0.065    0.007
word        spectral-signature     0.252  0.048    0.368    0.050    0.005
word        sae                    0.032  0.032    0.100    0.010    0.001
word        nla                    1.000  0.056    0.665    0.153    0.018
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  0.992  0.000    1.000    1.000    1.000
word        spectral-signature     0.356  0.048    0.452    0.070    0.007
word        onion                  0.192  0.040    0.348    0.046    0.005
word        logit-lens             0.048  0.048    0.100    0.010    0.001
word        sae                    0.040  0.024    0.156    0.017    0.002
word        nla                    1.000  0.080    0.581    0.112    0.012
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  1.000  0.000    1.000    1.000    1.000
word        onion                  0.256  0.048    0.372    0.051    0.005
word        sae                    0.200  0.032    0.410    0.059    0.006
word        spectral-signature     0.160  0.048    0.270    0.033    0.003
word        logit-lens             0.000  0.104    0.000    0.000    0.000
word        nla                    1.000  0.116    0.489    0.080    0.009
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  1.000  0.000    1.000    1.000    1.000
word        logit-lens             0.972  0.076    0.587    0.114    0.013
word        onion                  0.464  0.092    0.359    0.048    0.005
word        spectral-signature     0.144  0.048    0.250    0.029    0.003
word        sae                    0.020  0.020    0.100    0.010    0.001
word        nla                    1.000  0.032    0.776    0.240    0.030
word        hidden-svd             1.000  0.048    0.698    0.174    0.020
word        activation-clustering  1.000  0.000    1.000    1.000    1.000
word        logit-lens             0.968  0.080    0.573    0.109    0.012
word        onion                  0.632  0.080    0.467    0.074    0.008
word        spectral-signature     0.176  0.048    0.289    0.036    0.004
word        sae                    0.004  0.004    0.100    0.010    0.001
phrase      nla                    1.000  0.060    0.649    0.144    0.016
phrase      hidden-svd             0.064  0.048    0.129    0.013    0.001
phrase      sae                    0.024  0.024    0.100    0.010    0.001
phrase      onion                  0.004  0.028    0.016    0.001    0.000
phrase      logit-lens             0.000  0.092    0.000    0.000    0.000
phrase      spectral-signature     0.000  0.048    0.000    0.000    0.000
phrase      activation-clustering  0.000  0.048    0.000    0.000    0.000
phrase      nla                    0.996  0.084    0.568    0.107    0.012
phrase      hidden-svd             0.120  0.048    0.217    0.025    0.002
phrase      sae                    0.016  0.012    0.129    0.013    0.001
phrase      onion                  0.008  0.040    0.022    0.002    0.000
phrase      logit-lens             0.000  0.048    0.000    0.000    0.000
phrase      spectral-signature     0.000  0.048    0.000    0.000    0.000
phrase      activation-clustering  0.000  0.048    0.000    0.000    0.000
phrase      nla                    1.000  0.116    0.489    0.080    0.009
phrase      onion                  0.036  0.108    0.036    0.003    0.000
phrase      sae                    0.024  0.008    0.250    0.029    0.003
phrase      hidden-svd             0.004  0.048    0.009    0.001    0.000
phrase      logit-lens             0.000  0.048    0.000    0.000    0.000
phrase      spectral-signature     0.000  0.048    0.000    0.000    0.000
phrase      activation-clustering  0.000  0.048    0.000    0.000    0.000
syntactic   nla                    0.932  0.056    0.649    0.144    0.016
syntactic   hidden-svd             0.904  0.048    0.677    0.160    0.019
syntactic   onion                  0.040  0.068    0.061    0.006    0.001
syntactic   sae                    0.028  0.044    0.066    0.006    0.001
syntactic   logit-lens             0.024  0.048    0.053    0.005    0.001
syntactic   spectral-signature     0.024  0.048    0.053    0.005    0.001
syntactic   activation-clustering  0.000  0.048    0.000    0.000    0.000
syntactic   activation-clustering  0.968  0.024    0.818    0.289    0.039
syntactic   nla                    0.960  0.140    0.432    0.065    0.007
syntactic   hidden-svd             0.920  0.048    0.680    0.162    0.019
syntactic   logit-lens             0.308  0.176    0.163    0.017    0.002
syntactic   onion                  0.188  0.060    0.258    0.031    0.003
syntactic   spectral-signature     0.028  0.048    0.061    0.006    0.001
syntactic   sae                    0.012  0.052    0.025    0.002    0.000
syntactic   hidden-svd             0.980  0.048    0.694    0.171    0.020
syntactic   activation-clustering  0.972  0.024    0.818    0.290    0.039
syntactic   nla                    0.932  0.044    0.702    0.176    0.021
syntactic   logit-lens             0.524  0.032    0.645    0.142    0.016
syntactic   onion                  0.084  0.076    0.109    0.011    0.001
syntactic   spectral-signature     0.020  0.048    0.044    0.004    0.000
syntactic   sae                    0.008  0.024    0.036    0.003    0.000
style       nla                    0.924  0.072    0.588    0.115    0.013
style       hidden-svd             0.908  0.048    0.678    0.160    0.019
style       activation-clustering  0.900  0.012    0.893    0.431    0.070
style       logit-lens             0.208  0.112    0.171    0.018    0.002
style       spectral-signature     0.148  0.048    0.255    0.030    0.003
style       onion                  0.084  0.068    0.121    0.012    0.001
style       sae                    0.000  0.016    0.000    0.000    0.000
style       nla                    0.900  0.072    0.581    0.112    0.012
style       hidden-svd             0.900  0.048    0.676    0.159    0.018
style       activation-clustering  0.872  0.008    0.924    0.524    0.098
style       logit-lens             0.280  0.096    0.245    0.029    0.003
style       spectral-signature     0.120  0.048    0.217    0.025    0.002
style       onion                  0.100  0.048    0.188    0.021    0.002
style       sae                    0.000  0.000      n/a      n/a      n/a
style       nla                    0.936  0.100    0.510    0.086    0.009
style       hidden-svd             0.908  0.048    0.678    0.160    0.019
style       activation-clustering  0.872  0.004    0.960    0.688    0.179
style       onion                  0.076  0.036    0.190    0.021    0.002
style       spectral-signature     0.072  0.048    0.143    0.015    0.001
style       logit-lens             0.048  0.080    0.062    0.006    0.001
style       sae                    0.000  0.016    0.000    0.000    0.000

A detector at TPR 1.00 and FPR 0.05 gives precision 0.169 at a 1% base rate:
83 of every 100 alarms are false. That is the number an analyst experiences.

==========================================================================================================================
NOTES ON TABLE B
==========================================================================================================================
(a) cleanFT+trig = flag rate on TRIGGERED inputs to the CLEAN control model.
    Close to det.acc means the detector found the trigger TEXT, not the backdoor.
(c) floor        = flag rate on clean inputs to the clean model; should sit at the
    alarm budget, and is what makes a 'NOT POISONED' verdict measurable.
lift             = PR-AUC / base rate. At a 50% base rate a useless detector scores 1.00.

READING det.acc AGAINST ROC. det.acc is an *operating point* at the alarm
budget; ROC is threshold-free. A large gap between them is itself a result:
  high ROC, low det.acc  -> the detector ranks the poison above the clean but
     the two distributions overlap in the tail, so no threshold at this budget
     separates them. Raising the budget would buy recall at a precision cost.
  ROC far below 0.5      -> the detector is INVERTED: it scores poisoned inputs
     lower than clean. That is a confident wrong answer, not a weak one, and it
     usually means an assumption behind the method does not hold here.

============================================================================================================
TABLE C — STEALTH.  Detection accuracy as the trigger becomes invisible to input inspection.
============================================================================================================
detector                     word      phrase   syntactic       style
---------------------------------------------------------------------
nla                         0.999       0.999       0.941       0.920
hidden-svd                  1.000       0.063       0.935       0.905
activation-clustering       0.997       0.000       0.647       0.881
logit-lens                  0.702       0.000       0.285       0.179
onion                       0.432       0.016       0.104       0.087
spectral-signature          0.164       0.000       0.024       0.113
sae                         0.051       0.021       0.016       0.000

  word       rare token — an outlier word, trivially greppable
  phrase     fluent sentence — no perplexity signal, but a fixed exact string
  syntactic  rare parse template — every word ordinary
  style      Bible-style rewrite — perplexity *below* normal text

Averaged over seeds. A detector that reads internal representations should hold
across this row; one that reads the input text should fall off to the right.

============================================================================================================
TABLE D — POISON RATE.  Where detection collapses (word trigger).
============================================================================================================
detector                   0.2%      0.5%        1%        2%        3%        5%       10%
-------------------------------------------------------------------------------------------
activation-clustering         —         —     0.992     0.996     1.000     0.997     1.000
hidden-svd                    —         —     1.000     1.000     1.000     1.000     1.000
logit-lens                    —         —     0.984     0.960     0.984     0.340     0.968
nla                           —         —     0.992     1.000     1.000     1.000     1.000
onion                         —         —     0.464     0.548     0.468     0.304     0.632
sae                           —         —     0.020     0.044     0.032     0.087     0.004
spectral-signature            —         —     0.040     0.020     0.252     0.220     0.176
attack fired?                NO        NO       yes       yes       yes       yes       yes

A '—' is a rate at which the attack itself did not install, not a detector failure.

18 runs, 16 fired.
```
