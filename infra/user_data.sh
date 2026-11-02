#!/bin/bash
# EC2 '사용자 데이터'에 그대로 붙여 넣는 초기 설정 (Amazon Linux 2023)
#   Docker + compose 설치, 배포 폴더 준비, 재학습 파이프라인을 매일 04:00 KST 에 실행하는 타이머 등록
set -euxo pipefail
dnf install -y docker
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
install -d -o ec2-user /opt/mlopspr  # 배포 워크플로가 SSH(ec2-user)로 compose.yaml/.env 를 씀

cat > /etc/systemd/system/mlopspr-pipeline.service <<'UNIT'
[Unit]
Description=mlopspr retrain pipeline (crawl -> labels -> retrain)
After=docker.service
[Service]
Type=oneshot
WorkingDirectory=/opt/mlopspr
ExecStart=/usr/bin/docker compose run --rm pipeline
UNIT
cat > /etc/systemd/system/mlopspr-pipeline.timer <<'UNIT'
[Timer]
# 매일 04:00 KST
OnCalendar=*-*-* 19:00:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now mlopspr-pipeline.timer
