===========================================
고영과제
- for paper development, **Multi-target regression via input space expansion: treating targets as inputs**: discuss multi-label regression tasks (관련 dataset: MTR "mulan" dataset: http://mulan.sourceforge.net/format.html)
  - this is relevant to multivariate regression for reflow oven ==> not the main portion of this project
  - our goal is to optimize multiregression task according to an objective *관련 논문 찾아보기*
  - another BO paper: Bayesian Optimisation over Multiple Continuous and Categorical Inputs: https://arxiv.org/pdf/1906.08878.pdf


===========================================
1.6.2020
- 고영 NP 2-D input에 대해 single mean, variance 잘 나오게 만들기
- 고영 NP module을 사용하는 acquisition function modification
- 고영: task-incremental learning?
- 회의
  - Reflow oven: chip별로 하나씩 PRE, PRE-SPI 를 input으로 하는 것을 만들어서 test **done**

  - PRE, PRE-SPI (2)
  - Does normalizing by the chip length help predict better? normalize by chip length (3)
  *PRE-SPI*
  - Reflow oven에서 chip별로 모델을 불러와서 PRE, PRE-SPI 를 넣어서 post의 변화를 관측
  - * 여기서 sample 된 candidate(PRE-SPI)을 fix하려면 SPI값을 같은 칩에서 임의로 랜덤샘플

- 
- ai 28 FSOD model 관련 계획서 contribute
  1. use meta-learner as prior knowledge to guide each specific FSL task
    - lack of data environment: FSL / data augmentation
    - how to generalize feature maps for each class and store that in memory?
  2. incremental zero-shot learning?
  3. class incremental learning?
  4. online learning?


1.8.2020
- (1) 현재 reflow oven simulation한거를 잘 보려면 수치화된 post-aoi regression 에 대한 mean, variance 가 필요하다 (against random inputs)
- rf regressor를 조금 덜 train 되게 보이려면...? -> tree, depth갯수를 줄이자
- (2) input dimension을 늘려서 해보는 방식: PRE / PRE-SPI, SPI, SPI_VOLUME 등
- !가정하는 사실: mounter에서 noise를 배제하고, candidate으로 샘플한 value를 바로 넣는다
>> other
- reflow oven 이후로 만 판단하므로 reflow oven에 대해 dependency is high; let's see how the candidates relate to the actual data by plotting
1. change imputation code to include SPI L,W PRE L,W POST L,W, SPI_VOLUME_MEAN, SPI_VOLUME_DIFF
2. change RF regressor model
3. change dataset 