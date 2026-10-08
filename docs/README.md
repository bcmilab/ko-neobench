# KoNeoBench Project Page

KoNeoBench의 영문 학술 프로젝트 페이지입니다. 사용자 지정 [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)을 바탕으로 구성했습니다.

## 확인 방법

압축을 해제한 뒤 **index.html**을 브라우저에서 열면 됩니다. 별도의 설치, 빌드, API 키가 필요하지 않습니다. 그림, 스타일시트와 스크립트는 모두 폴더 안에 포함되어 있습니다. 논문·GitHub 등 외부 링크에는 인터넷 연결이 필요합니다.

함께 전달된 **KoNeoBench_preview.html**은 같은 페이지의 이미지·스타일·스크립트를 한 파일에 담은 검토용 파일입니다. 실제 수정에는 이 소스 폴더를 사용하세요. 브라우저의 보안 정책에 따라 클립보드 복사가 제한될 수 있으며, 그 경우 직접 복사하거나 `.bib` 파일을 내려받을 수 있습니다.

## 구성

- `index.html`: 영문 본문, 저자·소속, 수치, 링크, 메타데이터, BibTeX
- `static/css/bulma.min.css`: 템플릿에 포함된 Bulma 0.9.1
- `static/css/template.css`: 원본 템플릿의 기본 스타일
- `static/css/index.css`: KoNeoBench 초록색 테마와 반응형 구성
- `static/js/index.js`: 그림 확대, 정답 확인, 인용 복사, 상단 이동
- `static/images/`: 원 논문의 Figure 2, 3, 4, 6(b)와 favicon
- `static/files/koneobench.bib`: 내려받기용 arXiv 인용
- `REVIEW_NOTES.md`: 원문 대응, 보존한 불일치, 검증 범위
- `THIRD_PARTY_NOTICES.md`: 템플릿 및 외부 자산 출처
- `scripts/build_preview.py`: 검토용 단일 HTML 재생성

## 수정 위치

| 변경 사항 | 위치 |
|---|---|
| 저자, 소속, 연락처, 학회 표기 | `index.html`의 hero 및 footer |
| 과제 예시와 핵심 결과 | `index.html`의 tasks 및 findings |
| Task 3 정답 피드백 | `static/js/index.js` |
| 색상, 폭, 간격, 모바일 배치 | `static/css/index.css` |
| 인용 정보 | HTML의 BibTeX와 `static/files/koneobench.bib` 함께 변경 |
| 논문 그림 | `static/images/`의 해당 PNG/WebP 및 HTML alt/caption |

## 독립적인 정적 사이트

이 폴더는 연구 코드 저장소와 별개로 작성되었습니다. 모든 로컬 링크는 상대 경로이며, 루트 경로나 하위 경로에서 정적 호스팅할 수 있습니다. GitHub 저장소 생성, 연구 저장소 변경, GitHub Pages 활성화, 공개 배포는 수행하지 않았습니다. `.nojekyll`은 향후 GitHub Pages에서 그대로 사용할 수 있도록 포함했습니다.

배포 주소를 확정한 뒤에는 `canonical` 및 `og:url` 메타데이터를 추가할 수 있습니다. 현재는 미정 주소를 넣지 않았습니다. Google Analytics, 원격 폰트, 추적 스크립트, API 호출은 포함하지 않았습니다.

## 페이지 다시 묶기

Python 3으로 실행합니다. 추가 Python 패키지는 필요하지 않습니다.

```sh
python3 scripts/build_preview.py --output KoNeoBench_preview.html
```

## 확인 범위

원문 수치·예시·저자 정보, 로컬 자산 참조, 내부 앵커, BibTeX 일치, JavaScript 구문과 정적 구조를 확인했습니다. 현재 실행 환경에서는 브라우저 기반 시각 검증을 수행할 수 없어, 데스크톱·모바일의 실제 렌더링과 실제 브라우저에서의 클릭 동작은 별도 확인이 필요합니다.
