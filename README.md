# RAG Multimodal

여기에서는 Multimodal LLM을 이용하여 RAG를 구현합니다.

## 설치

### 사전 요구 사항

- Python 3.x
- AWS CLI 자격 증명이 구성된 상태 (`aws configure` 또는 환경 변수)
- `pip install -r requirements.txt`

### 인프라 배포

프로젝트 루트에서 installer를 실행합니다.

```bash
python3 installer.py
```

installer는 다음 리소스를 생성·구성합니다.

- S3 버킷 (`docs/` 프리픽스)
- Amazon OpenSearch Service 관리형 도메인 (`rag-multimodal`)
- CloudFront 배포
- **lambda-s3-event-manager**: S3 `docs/` PDF **삭제** 시 `metadata/*.metadata.json`의 `ids`로 OpenSearch 벡터 삭제 (IAM 역할 포함)

설치가 끝나면 `application/config.json`이 갱신됩니다.

### OpenSearch Dashboards (브라우저 접속)

installer는 OpenSearch **Fine-grained access control(FGAC)** 을 활성화하여 브라우저에서 Dashboards에 로그인할 수 있게 합니다.

| 항목 | 값 |
|------|-----|
| 사용자명 | `admin` (고정) |
| 비밀번호 | 설치 시 터미널에서 직접 입력 (두 번 확인), `application/config.json`에 저장 |

비밀번호 규칙 (AWS OpenSearch FGAC): 8~128자, 대문자·소문자·숫자 각 1자 이상.

FGAC가 이미 켜진 도메인을 재설치할 때는 비밀번호 입력을 건너뛰며, `config.json`에 기존 `managed_opensearch_dashboards_password`가 있으면 그대로 유지합니다.

installer는 FGAC 활성화 후 **도메인 액세스 정책**을 갱신합니다(IAM root + Dashboards용 요청 허용). FGAC 마이그레이션 모드(`AnonymousAuthEnabled`)가 켜져 있으면 먼저 끈 뒤 정책을 적용합니다. 실제 권한은 FGAC가 검사합니다.

#### 설치 후 접속

`config.json`의 `managed_opensearch_dashboards_url`로 접속합니다 (예: `https://<domain-endpoint>/_dashboards`).

- **Username:** `managed_opensearch_dashboards_user` (`admin`)
- **Password:** `managed_opensearch_dashboards_password`

브라우저에서 URL만 열면 Dashboards 로그인 화면(HTTP 302)으로 이동합니다. `admin` / 설치 시 비밀번호로 로그인하세요. IAM SigV4 RAG API 호출은 계정 root Principal 정책으로 계속 동작합니다.

#### config.json 관련 필드

| 필드 | 설명 |
|------|------|
| `managed_opensearch_url` | OpenSearch API 엔드포인트 |
| `managed_opensearch_dashboards_url` | Dashboards URL (`/_dashboards`) |
| `managed_opensearch_dashboards_user` | Dashboards 로그인 사용자명 (`admin`) |
| `managed_opensearch_dashboards_password` | Dashboards 로그인 비밀번호 (설치 시 입력) |
| `s3_docs_prefix` | S3 이벤트 대상 프리픽스 (`docs/`) |
| `lambda_s3_event_manager_arn` | S3 이벤트 처리 Lambda ARN |

`application/config.json`은 `.gitignore`에 포함되어 있으므로 비밀번호가 저장소에 커밋되지 않습니다.

### 인프라 삭제

```bash
python3 uninstaller.py
```
