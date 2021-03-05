#### self-alignment
목적
- 실제 MOM4차데이터의 분포를 잘 따라가는 (mimic) regressor 모델

진행상황
- 현재 활용하고 있는 모델: random forest regressor of width 50 and depth 100
- against a test subset of 200 samples (from actual data), it yields a score of 0.71

#### surrogate model
목적
- iteration 마다 N개의 새로운 샘플데이터를 추가하여 모델 재생성후 재학습 (재학습시간=다음데이터가 나오기 까지(Post-AOI단계까지)의 대기시간이라고 볼 수 있다)

진행상황
- 2d의 같은 경우, 한 이미지의 전체픽셀을 (예: 28x28=784) 전체 샘플로 볼 수 있는데, MOM4차 데이터의 경우 전체데이터를 분할해서 활용시, Neural process자체에서 분할하는 방법으로 인해 샘플사이즈에 따른어려움 (너무작아짐)이 있을 수 있으므로, batch size를 1로 해서 전체 데이터를 한꺼번에 프로세싱하는 방법 고려중

#### acquisition function
목적
- GP/NP모델을 기반으로, search space내에서 explore(initial conditions생성단계) 한후, output을 최소/최대화하는 샘플 선정

진행상황
- UCB 를 기본알고리즘으로 테스트 할 계획
- BoTorch 에서 GP에 특화된 Acquisition function (분포 등)에서 NP모델을 다시 점검하고, 기존에 proposed된 loss function이 2d image completion task등 에서 좋은 성능을 이끌지 못함으로 인해 모델-optimization process간 시행착오와 underlying process를 전반적으로 훑고 개선하는중
- 현재 search space 를 고려할때,데이터의 구조적인 문제 발생...원인파악중
