# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 이 리포

FC_Hackathon. **무엇을 만들지 아직 정하지 않았다.** 스택도 미정.
이 리포는 지금 인계 프로토콜만 깔려 있는 상태다.

## 인계 프로토콜 (가장 중요)

**클로드코드와 커서가 이 폴더 하나를 교대로 쓴다.** 두 도구는 서로의 대화를 볼 수 없다.
`HANDOFF.md`가 유일한 인계 채널이다.

1. **세션 시작 시 `HANDOFF.md`를 먼저 읽는다.** 그다음 `git pull`, `git log --oneline -5`로 대조.
   문서와 커밋이 어긋나면 커밋이 진실이다.
2. 되돌리기 어려운 판단을 했으면 `HANDOFF.md`의 "결정 로그"에 한 줄 (버린 대안 포함).
3. **끝낼 때 `HANDOFF.md`의 "현재 상태"를 덮어쓰고 "세션 로그"에 항목 1개를 추가한 뒤,
   코드 변경과 같은 커밋에 담아 push한다.** 도구를 바꾸기 직전에는 반드시 여기까지.
4. 커밋 제목 앞에 태그 `[CC]`. (커서는 `[CU]`)
5. 두 도구를 **동시에** 열지 않는다. 한쪽을 닫고 다른 쪽을 연다.

전체 절차와 세션 로그 양식은 `HANDOFF.md` §프로토콜에 있다.
(`HANDOFF.md`는 append-growing 문서이므로 이 파일에서 `@import` 하지 않는다. 경로 참조만 한다.)

## 빌드 / 테스트 / 린트

(미정 - 스택 결정 시 여기 채울 것)

## 커서 쪽 대응 파일

`.cursor/rules/handoff.mdc`가 같은 프로토콜을 담고 있다.
**한쪽을 고치면 다른 쪽도 같은 자리에서 고친다.** 두 파일이 다른 말을 하기 시작하면
프로토콜이 아니라 분쟁거리가 된다.
