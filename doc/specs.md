# specs
- optimizer using SGD
- NP, GP reinitialized at each training step
- context size incremented for every candidate (for NP)
- at every epoch, context and target are set differently, but of equal sizes
- mounter는 계속 PRE를 return해야하지만, POST는 받는데 딜레이가 있다
- 
# todo
- MOUNTER: noise level 어느정도 줘야할지 데이터에서 찾아보기. 현재 임의로 120 
준 상태
- REFLOW OVEN: 중심 origin에서 상대적으로 많이 떨어진 PRE에서 더 잘 모이는지 등 비교


# questions
- SPI - MOUNTER 사이 시간차이? --> 충분하다면 SPI VOLUME에 따라 mounter input (pre) 추가보정 가능
- 쓰고 있는 mounter의 오차범위가 어느정도인지?
- reflow oven 시뮬레이션에 대해 파악된 정보가 있으면 비슷하게 그려보는데 도움이 될듯

# ideas
- INPUTS: (1) PRE (2) PRE-SPI (3) PRE | SPI | SPI_VOL (4) PRE-SPI | SPI_VOL
- 어떻게 보면, 지금 같은 모델을 여러번 학습시키는 것도, 일종의 joint training이라고 볼 수 있다; 하지만 overfitting이라는 단점
- 
-   

