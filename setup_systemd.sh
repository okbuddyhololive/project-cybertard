sudo bash setup.sh
sudo cp scripts/discord.service /etc/systemd/system/discord.service
sudo systemctl daemon-reload
sudo systemctl enable discord.service
sudo systemctl start discord.service
