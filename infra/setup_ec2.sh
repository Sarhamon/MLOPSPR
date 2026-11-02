#!/usr/bin/env bash
# AWS CLI 로 배포 리소스를 만드는 스크립트 (IAM 역할을 만들지 않음 - 학교 계정용). 설명은 infra/README.md
#
#   bash infra/setup_ec2.sh          # 저장소 루트에서 실행 (Windows 는 Git Bash)
#
# 만드는 것: ECR 리포지토리, 키 페어(~/.ssh/mlopspr.pem), 보안 그룹(80, 22), EC2 1대, 탄력적 IP
# 이미 있는 것은 다시 만들지 않으므로 중간에 실패하면 고친 뒤 다시 실행하면 된다.
# 지우기: bash infra/teardown_ec2.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash 가 /aws/... 같은 인자를 Windows 경로로 바꾸지 않도록

export AWS_REGION="${AWS_REGION:-ap-northeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
NAME="${NAME:-mlopspr}"
ECR_REPO="${ECR_REPO:-plus/mlops}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.small}"
HTTP_CIDR="${HTTP_CIDR:-0.0.0.0/0}"  # 서비스(80)에 접속할 수 있는 범위
SSH_CIDR="${SSH_CIDR:-0.0.0.0/0}"    # GitHub Actions 러너 IP 가 고정이 아니라 기본은 전체 (키로만 로그인)
KEY_FILE="$HOME/.ssh/$NAME.pem"

q() { aws "$@" --output text | tr -d '\r'; }  # Windows 용 aws.exe 출력의 \r 제거

command -v aws >/dev/null || { echo "AWS CLI 가 없습니다. 설치 후 aws configure 를 실행하세요."; exit 1; }
ACCOUNT_ID=$(q sts get-caller-identity --query Account)  # 자격 증명이 없으면 여기서 멈춤
echo "계정 $ACCOUNT_ID / 리전 $AWS_REGION"

# 1) ECR
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO" --image-scanning-configuration scanOnPush=true >/dev/null
fi
echo "ECR: $ECR_REPO"

# 2) 키 페어 (개인 키는 이때 한 번만 받을 수 있음)
if ! aws ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then
  mkdir -p "$(dirname "$KEY_FILE")"
  q ec2 create-key-pair --key-name "$NAME" --key-type rsa --query KeyMaterial > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
  echo "키 페어: $NAME (개인 키 저장: $KEY_FILE)"
else
  echo "키 페어: $NAME (이미 있음 - 개인 키는 처음 만들 때 저장한 파일을 사용)"
fi

# 3) 보안 그룹 (기본 VPC)
SG_ID=$(q ec2 describe-security-groups --filters "Name=group-name,Values=$NAME-web" --query 'SecurityGroups[0].GroupId')
if [ "$SG_ID" = "None" ]; then
  SG_ID=$(q ec2 create-security-group --group-name "$NAME-web" --description "$NAME web+ssh" --query GroupId)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr "$HTTP_CIDR" >/dev/null
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$SSH_CIDR" >/dev/null
fi
echo "보안 그룹: $SG_ID (80 <- $HTTP_CIDR, 22 <- $SSH_CIDR)"

# 4) EC2 (Amazon Linux 2023, 사용자 데이터 = infra/user_data.sh)
INSTANCE_ID=$(q ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
  "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[0].Instances[0].InstanceId')
if [ "$INSTANCE_ID" = "None" ]; then
  AMI_ID=$(q ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value 2>/dev/null) || AMI_ID=""
  if [ -z "$AMI_ID" ]; then  # SSM 읽기 권한이 없는 계정이면 이미지 목록에서 최신 AL2023 을 고름
    AMI_ID=$(q ec2 describe-images --owners amazon \
      --filters "Name=name,Values=al2023-ami-2023.*-kernel-*-x86_64" "Name=state,Values=available" \
      --query 'sort_by(Images,&CreationDate)[-1].ImageId')
  fi
  USER_DATA=$(mktemp)
  tr -d '\r' < "$(dirname "$0")/user_data.sh" > "$USER_DATA"  # Windows 체크아웃의 CRLF 가 남으면 스크립트가 실행되지 않음
  INSTANCE_ID=$(q ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
    --key-name "$NAME" --security-group-ids "$SG_ID" \
    --metadata-options HttpTokens=required \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
    --user-data "file://$USER_DATA" --query 'Instances[0].InstanceId')
  rm -f "$USER_DATA"
  echo "EC2 생성: $INSTANCE_ID ($INSTANCE_TYPE, $AMI_ID) - 실행 대기 중..."
fi
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
echo "EC2: $INSTANCE_ID"

# 5) 탄력적 IP
ALLOC_ID=$(q ec2 describe-addresses --filters "Name=tag:Name,Values=$NAME" --query 'Addresses[0].AllocationId')
if [ "$ALLOC_ID" = "None" ]; then
  ALLOC_ID=$(q ec2 allocate-address --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" --query AllocationId)
fi
aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" >/dev/null
PUBLIC_IP=$(q ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].PublicIp')
echo "탄력적 IP: $PUBLIC_IP"

cat <<EOF

=== 완료. GitHub 저장소 Settings > Secrets and variables > Actions 에 등록하세요 ===
[Variables]
  AWS_REGION     = $AWS_REGION
  ECR_REPOSITORY = $ECR_REPO
  EC2_HOST       = $PUBLIC_IP
  SERVICE_URL    = http://$PUBLIC_IP
[Secrets]
  AWS_ACCESS_KEY_ID     = aws configure 에 넣은 액세스 키 ID
  AWS_SECRET_ACCESS_KEY = aws configure 에 넣은 비밀 액세스 키
  EC2_SSH_KEY           = $KEY_FILE 파일 내용 전체

EC2 초기 설정(Docker 설치)은 2~3분 걸립니다. 확인:
  ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'docker compose version && systemctl list-timers $NAME-pipeline.timer'
EOF
