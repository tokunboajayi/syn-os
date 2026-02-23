#!/bin/bash

# Enable services
systemctl enable docker.service
systemctl enable NetworkManager.service
systemctl enable sshd.service

# Create synos user
useradd -m -G docker,wheel -s /bin/bash synos
echo "synos:synos" | chpasswd
echo "root:synos" | chpasswd

# Configure sudo
echo "%wheel ALL=(ALL) ALL" >> /etc/sudoers

# Set up Xorg
echo "exec /usr/local/bin/synos-kiosk.sh" > /home/synos/.xinitrc
chown synos:synos /home/synos/.xinitrc

# Clone Syn OS (Optimized for Live Environment)
# Ideally this would pull a pre-built release artifact
git clone https://github.com/olatunji/syn-os.git /opt/syn-os
chown -R synos:synos /opt/syn-os

# Pre-pull critical images if internet is available during build
# systemctl start docker
# docker pull postgres:15-alpine
# docker pull redis:7-alpine
