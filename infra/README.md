배포 구성 (EC2 + docker compose, IAM 역할 없이)
===============================================

학교 계정처럼 IAM 역할을 만들 수 없는 환경 기준. 필요한 권한은 본인 IAM 사용자의 액세스 키 하나.

구조
----
    GitHub (main push)
      └─ CI: pytest + 도커 이미지 빌드/스모크 테스트
           └─ 성공하면 Deploy (액세스 키로 AWS 로그인)
                ├─ 이미지 빌드 → ECR(plus/mlops) 푸시 (태그 = 커밋 SHA)
                ├─ SSH → EC2: ECR 로그인 토큰 전달, compose.yaml/.env 갱신, pull/up (app 만 교체)
                └─ 공개 주소로 smoke_test.py

    EC2 (Amazon Linux 2023, t3.small, 탄력적 IP)        사용자 브라우저
      /opt/mlopspr/compose.yaml, .env (IMAGE=...)              │ :80
      ├─ app 컨테이너 (8000 → 호스트 80) ◀────────────────────┘
      │     /data 의 모델/추천 후보가 바뀌면 다음 요청 때 다시 읽음
      ├─ pipeline 컨테이너 (systemd 타이머, 매일 04:00 KST, 끝나면 사라짐)
      │     크롤링 01 → 정답 03 → 검증/재학습/추천 후보 갱신 04 → /data 에 기록
      └─ 볼륨 data (/data): cars_processed.csv, price_model.joblib, recommend_pool.csv,
                            models/ (보관 모델, registry.jsonl), logs/ (요청 로그, 파이프라인 로그)

왜 이렇게 했나
- 앱은 이미 "파일이 바뀌면 재시작 없이 다시 읽기"로 만들어져 있어서, 같은 호스트에서 볼륨을
  공유하면 코드 변경 없이 재학습 결과가 바로 서비스에 반영된다.
- IAM 역할을 만들 수 없어 EC2 에 역할을 붙이지 못한다. 그래서 EC2 에는 AWS 키를 두지 않고,
  배포할 때 Actions 가 ECR 로그인 토큰(12시간 유효)을 받아 SSH 로 넘겨 준다.
  재학습 파이프라인은 이미 받아 둔 이미지를 쓰므로 ECR 접근이 필요 없다.

알아둘 점
- SSH(22)를 GitHub Actions 러너에서 접속할 수 있게 열어야 한다. 러너 IP 가 고정이 아니라
  0.0.0.0/0 으로 열게 되며, 비밀번호 로그인은 꺼져 있고 키 페어로만 접속된다.
- 액세스 키는 장기 키다. GitHub Secrets 에만 두고, 유출 의심 시 IAM 에서 즉시 비활성화한다.
- 볼륨은 처음 만들어질 때만 이미지의 초기 데이터(/data)로 채워진다. 이후 배포는 코드만 바꾸고
  볼륨의 모델/데이터(재학습 결과)는 그대로 둔다.
- HTTPS/도메인은 포함하지 않았다. 서버 1대라 배포 중 몇 초 끊길 수 있다.

처음 한 번: AWS CLI 로 만들기 (리전: 서울 ap-northeast-2)
---------------------------------------------------------
필요: AWS CLI v2, 본인 IAM 사용자의 액세스 키 (IAM > 사용자 > 본인 > 보안 자격 증명 > 액세스 키 만들기)

    aws configure                 # 액세스 키 ID / 비밀 키 / ap-northeast-2 / json
    bash infra/setup_ec2.sh       # 저장소 루트에서 (Windows 는 Git Bash)
    # 선택: HTTP_CIDR=<내 IP>/32 (서비스 접속 제한), INSTANCE_TYPE, ECR_REPO(기본 plus/mlops)

ECR, 키 페어(~/.ssh/mlopspr.pem), 보안 그룹(80, 22), EC2(사용자 데이터 = infra/user_data.sh),
탄력적 IP 를 만들고 GitHub 에 등록할 값을 출력한다. 이미 있는 리소스는 건너뛰므로 다시 실행해도 된다.
지우기: bash infra/teardown_ec2.sh  (ECR 까지: DELETE_ECR=1)

콘솔에서 만들 경우 (위 스크립트와 같은 구성)
-------------------------------------------
1) ECR 리포지토리: plus/mlops (프라이빗)
2) 키 페어: EC2 > 키 페어 > 생성, 이름 mlopspr, RSA, .pem 다운로드
3) 보안 그룹: mlopspr-web (기본 VPC)
   - 인바운드 HTTP 80  ← 0.0.0.0/0 (또는 내 IP)
   - 인바운드 SSH 22   ← 0.0.0.0/0 (GitHub Actions 배포용)
4) EC2 인스턴스: 이름 mlopspr, Amazon Linux 2023 (x86_64), t3.small, 키 페어 mlopspr,
   보안 그룹 mlopspr-web, 스토리지 20GB gp3, IAM 인스턴스 프로파일 없음,
   고급 세부 정보 > 사용자 데이터 = infra/user_data.sh 내용 그대로
5) 탄력적 IP: 할당 → 작업 > 주소 연결 → mlopspr 인스턴스
6) 액세스 키: IAM > 사용자 > 본인 > 보안 자격 증명 > 액세스 키 만들기 (사용 사례: 서드 파티 서비스)
   이 사용자에게 ECR 푸시 권한이 있어야 한다 (ECR 리포지토리를 만들 수 있었다면 보통 있음)

GitHub 설정 (저장소 Settings > Secrets and variables > Actions)
--------------------------------------------------------------
    Variables 탭
      AWS_REGION      = ap-northeast-2
      ECR_REPOSITORY  = plus/mlops
      EC2_HOST        = <탄력적 IP>
      SERVICE_URL     = http://<탄력적 IP>
      EC2_USER        = (선택, 기본 ec2-user)
    Secrets 탭
      AWS_ACCESS_KEY_ID     = 6) 의 액세스 키 ID
      AWS_SECRET_ACCESS_KEY = 6) 의 비밀 액세스 키
      EC2_SSH_KEY           = 2) 의 .pem 파일 내용 전체 (-----BEGIN ... END----- 포함)
EC2_HOST 가 없으면 Deploy 는 건너뛰므로, 등록 전 push 는 CI 만 돈다.

배포
----
- 자동: main 에 push(PR 머지) → CI 성공 → Deploy
- 수동: GitHub Actions > Deploy > Run workflow

서비스 확인
-----------
    python smoke_test.py http://<탄력적 IP>   # Deploy 마지막 단계와 같은 검사
    브라우저: http://<탄력적 IP>/  (화면),  http://<탄력적 IP>/docs (API 문서)

운영 점검
---------
    ssh -i mlopspr.pem ec2-user@<탄력적 IP>
    cd /opt/mlopspr
    sudo docker compose ps / logs app
    sudo docker compose run --rm pipeline                    # 파이프라인 지금 바로 실행
    systemctl list-timers mlopspr-pipeline.timer             # 다음 실행 시각
    sudo docker compose run --rm --entrypoint cat pipeline /data/models/registry.jsonl
    sudo cat /var/log/cloud-init-output.log                  # 사용자 데이터(초기 설정) 실행 결과

롤백
----
    Actions > Deploy 를 이전 커밋에서 다시 실행 (이미지 태그 = 커밋 SHA)
    모델 롤백: 볼륨의 models/model_<버전>.joblib 을 price_model.joblib 으로 복사하면 app 이 다시 읽음

정리 (과금 중지)
----------------
    bash infra/teardown_ec2.sh   # 또는 콘솔에서: EC2 종료 → 탄력적 IP 릴리스(연결 안 된 EIP 도 과금)
                                 #   → 보안 그룹/키 페어 삭제 → ECR 리포지토리 삭제
    이후 IAM 액세스 키 비활성화/삭제, GitHub Secrets 삭제
