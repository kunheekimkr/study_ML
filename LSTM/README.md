# Long Short-Term Memory

Neural Computation 9(8):1735-1780, 1997

## References

- [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Approach

- 기존 RNN 모델은 현재 시점으로부터 시간 격차가 커질수록 학습하는 정보를 계속 이어나가기 힘들어한다는 문제점을 가진다. 이는 RNN 의 장기 의존성 문제 (Long-Term Dependency)라고 하며, 신경망의 역전파 과정에서 신경망이 곱셈 연산을 기반으로 만들어져 있어 연산이 중첩되면 gradient가 발산하거나 수렴하기 때문에 발생하는 문제이다.

- 이를 해결하기 위해 LSTM 모델은 Cell State 와 Gate를 활용한다. Cell State는 모델과 수평으로 흐르는 흐름으로, 기존 입력 값의 정보를 뒤로 전달하는 역할을 수행한다. LSTM은 세 종류의 게이트를 통해 Cell State를 업데이트하여 저장할 정보와 버릴 정보를 관리한다.

## Model

![LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

LSTM의 전체적인 구조는 다음과 같다. x는 Input, h는 Output 이며, 위쪽의 흐름이 Cell State이다.

1. Forget Gate

![Forget Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

Forget Gate layer는 Cell State가 가진 정보 중 어떤 것을 버려야 될 지를 정하는 레이어이다. x와 h의 값이 시그모이드 레이어를 통해 Cell State에 전달된다.

2. Input Gate

![Input Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

Input Gate Layer는 새로운 정보 중 어떤 것을 Cell State에 저장할 것인지 정하는 레이어이다. 시그모이드 레이어가 어떤 값을 업데이트할지 정하고, tanh 레이어가 새로운 후보 값 벡터를 만든다.

3. Update Cell State

![Update Cell State](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

기존의 Cell State와 (1) 의 결과를 XOR 연산하여 버려야 될 정보를 버린 후, (2) 의 두 결과의 곱을 Cell State에 더해 주어 Cell State에 새로운 정보를 저장한다.

4. Output Gate

![Output Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

Output Gate Layer는 Output으로 출력할 값을 정하는 레이어이다. 시그모이드 레이어에 x와 h를 넣아 Cell State의 어느 부분을 Output으로 출력할 지 정한 후, tanh 레이어에 Cell State를 넣은 결과를 곱해 ㅒutput을 연산한다.
