# gk2aICT
저장소 위치: https://s3.console.aws.amazon.com/s3/buckets/60hz.data?prefix=kmipa/gk2a/&region=ap-northeast-2
주의사항: 파일/디렉토리 상 날짜 및 시간은 모두 UTC 기준이므로, 분석 및 API 서비스에서 감안하여 개발해야함

## install
```bash
# create conda env
$ conda create -n gk2a python==3.8.8 -y

# install dependencies
$ conda activate gk2a
$ conda install -c conda-forge opencv==4.5.5

# python dependencies
$ pip install -r requirements.txt
```

