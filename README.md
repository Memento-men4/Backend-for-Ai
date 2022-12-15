# 🧠 MEMENTO       ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
```
This is Memento's ML Backend  
(Memento's App Backend : https://github.com/Memento-men4/MEMENTO-Backend)
(Memento's Ai recognition :https://github.com/Memento-men4/MEMENTO-AI-recognition)
```
> About Memento > https://github.com/Memento-men4  
> Watch Demo video > https://www.youtube.com/watch?v=7lxRL_99KnQ

### 🖇 Architecture Design
<img width="489" alt="백엔드아키텍처" src="https://user-images.githubusercontent.com/91522259/207849988-2bc60dcf-9f8e-4987-8724-c1ddae513d67.png">  

### 🗄 ML Backend
Client : App server
```
기능 1 : App server로부터 사용자의 Recording 데이터를 받아 인공지능 모델을 실행시킨다.
기능 2 : 그 결과 만들어진 사용자 맞춤형 퀴즈들을 Main server에 반환한다.
```

### 📁 Api Documentation
Request
```
{
    "text1" : "오늘 3시 40분에 하늘이랑 점심으로 치즈돈까스랑 김치찌개를 먹었다.",
    "text2" : "나랑 아빠랑 8시 10분에 제일곱창으로 갔는데 사람이 너무 많아 대기시간이 너무 길어서 한시간 웨이팅하고 곱창을 먹을 수 있었다. 오랫동안 기다리게 해서 사장님이 죄송하다고 했다.",
    "text3" : "교수님이 오늘 석철이한테 과제를 내지 않았다고 혼을 내셨다. 내가봐도 교수님이 내신 과제가 너무 어려웠는데 교수님께서 다음번엔 과제를 쉽게 내주셨으면 좋겠다.",
    "text4" : "나 어제 늦게 집에 도착해서 세탁기 돌려야 하는 거 잊었어. 오늘은 꼭 세탁기 돌려야겠다. 나 오늘 또 늦게 도착하면 세희한테 빨래 해달라고 부탁해야겠다.",
    "text5" : "난 오늘 인공지능 및 응용 과목에서 RNN에 대해서 배웠다. 너무 어려웠지만 흥미로웠다. 방학에 더 많이 공부해야겠다고 생각했다.",
    "text6" : "내가 오늘 기분이 좋지 않아서 엄마에게 소리지르고 화를 냈다. 너무 속상했지만 엄마한테 미안했다. 그래서 내가 무지개 마카롱을 선물해드릴거다. 그리고 미안하다고 하고 엄마를 꼭 안아줄거다",
    "text7" : "오늘 정보시스템학과 동기들이랑 후배들이랑 선배들이랑 종강파티를 했다. 처음 보는 후배도 만났는데 너무 재밌었다. 다같이 노래방도 갔는데 내가 좋아하는 사람이 노래를 엄청 잘불러서 깜짝 놀랬다."
}
```

Response
```
{
    "text1" : "오늘 3시 40분에 하늘이랑 점심으로 먹은 음식은?",
    "text2" : "나랑 아빠랑 8시 10분에 갔던 가게의 이름은?",
    "text3" : "교수님이 오늘 과제를 내지 않았다고 혼낸 사람은?",
    "text4" : "오늘 늦게 도착하면 세희에게 부탁해야 하는 것은?",
    "text5" : "오늘 인공지능 및 응용 과목에서 무엇에 대해 배웠는가?",
    "text6" : "엄마에게 미안하다고 하고 선물해 드릴 것은?",
    "text7" : "동기들과 후배들과 선배들과 뒷풀이 한 장소는"
}
```


### 📝 Tech Blog
https://velog.io/@hanueleee/series/HYUSE  

### 🛠 Stacks

- 사용 기술 : Flask
- 배포 : AWS EC2
