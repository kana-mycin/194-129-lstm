function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt --no-check-certificate "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

mkdir Data_emnlp16
cd Data_emnlp16
gdrive_download 0B8GgebBFUM0IanRrTXhpd01hYTA 20NG.zip
gdrive_download 0B8GgebBFUM0ISXVSeVduYlpzU2M AGnews.zip
gdrive_download 0B8GgebBFUM0ITG96bVdOOVlDRG8 IMDB.zip
gdrive_download 0B8GgebBFUM0IQV9Fb05lWjZaeUU PMNIST.zip
unzip \*.zip