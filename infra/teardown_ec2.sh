#!/usr/bin/env bash
# setup_ec2.sh 가 만든 리소스를 지운다 (과금 중지). 볼륨의 재학습 결과/로그도 함께 사라진다.
#   bash infra/teardown_ec2.sh               # EC2, 탄력적 IP, 보안 그룹, 키 페어
#   DELETE_ECR=1 bash infra/teardown_ec2.sh  # + ECR 리포지토리(이미지 포함)
set -uo pipefail  # 이미 지워진 리소스가 있어도 나머지는 계속 지우도록 -e 는 쓰지 않음
export MSYS_NO_PATHCONV=1

export AWS_REGION="${AWS_REGION:-ap-northeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
NAME="${NAME:-mlopspr}"
ECR_REPO="${ECR_REPO:-plus/mlops}"

q() { aws "$@" --output text | tr -d '\r'; }

read -r -p "리전 $AWS_REGION 의 '$NAME' 리소스를 지웁니다$([ "${DELETE_ECR:-}" = 1 ] && echo " (ECR $ECR_REPO 포함)"). 계속할까요? [y/N] " yes
[ "$yes" = "y" ] || exit 1

for ALLOC_ID in $(q ec2 describe-addresses --filters "Name=tag:Name,Values=$NAME" --query 'Addresses[].AllocationId'); do
  ASSOC_ID=$(q ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].AssociationId')
  [ "$ASSOC_ID" != "None" ] && aws ec2 disassociate-address --association-id "$ASSOC_ID"
  aws ec2 release-address --allocation-id "$ALLOC_ID" && echo "탄력적 IP 릴리스: $ALLOC_ID"
done

INSTANCE_IDS=$(q ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
  "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId')
if [ -n "$INSTANCE_IDS" ]; then
  aws ec2 terminate-instances --instance-ids $INSTANCE_IDS >/dev/null
  echo "EC2 종료 대기: $INSTANCE_IDS"
  aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS
fi

SG_ID=$(q ec2 describe-security-groups --filters "Name=group-name,Values=$NAME-web" --query 'SecurityGroups[0].GroupId')
[ "$SG_ID" != "None" ] && aws ec2 delete-security-group --group-id "$SG_ID" && echo "보안 그룹 삭제: $SG_ID"
aws ec2 delete-key-pair --key-name "$NAME" && echo "키 페어 삭제: $NAME (로컬 ~/.ssh/$NAME.pem 은 직접 지우세요)"

if [ "${DELETE_ECR:-}" = 1 ]; then
  aws ecr delete-repository --repository-name "$ECR_REPO" --force >/dev/null && echo "ECR 삭제: $ECR_REPO"
fi
echo "완료. 액세스 키를 더 쓰지 않으면 IAM 에서 비활성화하고 GitHub Secrets 도 지우세요."
