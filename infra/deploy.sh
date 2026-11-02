#!/usr/bin/env bash
# 로컬 배포 스크립트 (AWS CLI): 이미지 빌드 → ECR 푸시 → SSH 로 EC2 의 app 교체 → 스모크 테스트
# .github/workflows/deploy.yml 과 같은 단계를 로컬 PC 에서 실행한다. 리소스는 infra/setup_ec2.sh 로 먼저 만든다.
#
#   AWS_PROFILE=mfa NAME=mlopspr-2024100161 bash infra/deploy.sh   # 저장소 루트에서 (Windows 는 Git Bash)
#
# 같은 환경에 배포됐는지 확인할 수 있도록 배포 전/후의 /health (모델·추천 후보 버전)와 이미지 태그를 출력한다.
# 볼륨(/data)은 그대로 두고 app 컨테이너만 새 이미지로 바꾸므로, 모델/추천 후보 버전은 배포 전과 같아야 정상이다.
set -euo pipefail
export MSYS_NO_PATHCONV=1

export AWS_REGION="${AWS_REGION:-ap-northeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
NAME="${NAME:-mlopspr}"
ECR_REPO="${ECR_REPO:-plus/mlops}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/$NAME.pem}"
EC2_USER="${EC2_USER:-ec2-user}"
PYTHON="${PYTHON:-$(command -v python3 >/dev/null && echo python3 || echo python)}"

q() { aws "$@" --output text | tr -d '\r'; }  # Windows 용 aws.exe 출력의 \r 제거
export PYTHONIOENCODING=utf-8  # Windows 콘솔에서 한글 출력이 깨지지 않도록
# 같은 환경인지 비교할 값만 출력: 모델/추천 후보 버전(볼륨에 있음), 후보 건수, 앱이 파일을 읽은 시각
health() { "$PYTHON" -c "import urllib.request,json,sys; h=json.load(urllib.request.urlopen(sys.argv[1] + '/health', timeout=5)); print('model_version=%(model_version)s pool_version=%(pool_version)s pool_rows=%(pool_rows)s loaded_at=%(loaded_at)s' % h)" "$1" 2>/dev/null || echo "(응답 없음)"; }

cd "$(dirname "$0")/.."
for c in aws docker ssh scp git; do command -v "$c" >/dev/null || { echo "$c 명령이 없습니다."; exit 1; }; done
[ -f "$KEY_FILE" ] || { echo "SSH 키가 없습니다: $KEY_FILE (setup_ec2.sh 가 만든 키 또는 KEY_FILE=경로)"; exit 1; }

# 배포 대상: EC2_HOST 를 주지 않으면 setup_ec2.sh 가 붙인 이름 태그로 찾는다 (탄력적 IP → 인스턴스 퍼블릭 IP 순)
if [ -z "${EC2_HOST:-}" ]; then
  EC2_HOST=$(q ec2 describe-addresses --filters "Name=tag:Name,Values=$NAME" --query 'Addresses[0].PublicIp')
  if [ "$EC2_HOST" = "None" ]; then
    EC2_HOST=$(q ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=running" \
      --query 'Reservations[0].Instances[0].PublicIpAddress')
  fi
  [ "$EC2_HOST" != "None" ] || { echo "'$NAME' 이름의 실행 중인 EC2 를 찾지 못했습니다. EC2_HOST=<IP> 로 지정하세요."; exit 1; }
fi
URL="http://$EC2_HOST"

# 이미지 태그 = 커밋 SHA. 커밋하지 않은 변경이 있으면 -dirty-시각 을 붙여 구분
TAG=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then TAG="$TAG-dirty-$(date +%m%d%H%M)"; fi
ACCOUNT_ID=$(q sts get-caller-identity --query Account)
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE="$REGISTRY/$ECR_REPO:$TAG"

echo "=== 배포 대상: $EC2_USER@$EC2_HOST / 이미지 $IMAGE"
echo "--- 배포 전 상태"
echo "health: $(health "$URL")"

echo "--- 1) 이미지 빌드 + ECR 푸시"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

echo "--- 2) EC2 에 배포 (SSH)"
OPTS=(-i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
scp "${OPTS[@]}" compose.yaml "$EC2_USER@$EC2_HOST:/opt/mlopspr/compose.yaml"
aws ecr get-login-password | ssh "${OPTS[@]}" "$EC2_USER@$EC2_HOST" "sudo docker login --username AWS --password-stdin $REGISTRY"
ssh "${OPTS[@]}" "$EC2_USER@$EC2_HOST" "set -e; cd /opt/mlopspr
  echo \"이전 이미지: \$(grep -s IMAGE= .env || echo 없음)\"
  printf 'IMAGE=%s\nPORT=80\n' '$IMAGE' > .env
  sudo docker compose pull app
  sudo docker compose up -d --wait app
  sudo docker image prune -af --filter until=168h >/dev/null
  sudo docker compose ps"

echo "--- 3) 스모크 테스트"
for i in 1 2 3 4 5 6; do
  "$PYTHON" smoke_test.py "$URL" && break
  [ "$i" = 6 ] && exit 1
  echo "재시도 대기 ($i)"; sleep 10
done

echo "--- 배포 후 상태"
echo "health: $(health "$URL")"
echo "=== 배포 완료: $URL  (이미지 $TAG)"
