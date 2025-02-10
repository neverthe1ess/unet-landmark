import csv
import os

# CSV 파일 경로 및 이미지가 들어 있는 폴더 경로 지정
CSV_PATH = 'project-5-at-2025-02-09-15-17-9564c34a.csv'
IMAGE_FOLDER = '/Users/neverthe1ess/PycharmProjects/youtube-cnn-002-pytorch-unet/raw'

# 1. CSV 파일을 읽어 id -> file_name 매핑 딕셔너리 생성
id_to_file_name = {}
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    # CSV의 헤더가 "annotation_id, id, ab_path, file_name"인 경우 DictReader 사용
    reader = csv.DictReader(f)
    for row in reader:
        # row['id'] : 예) "1680"
        # row['file_name'] : 예) "10.jpg"
        id_to_file_name[row['id']] = row['file_name']

# 2. 이미지 폴더 내 파일들에 대해 이름 변경
for file_name in os.listdir(IMAGE_FOLDER):
    # "task-1680.jpg"처럼 "task-"로 시작하고 ".jpg"로 끝나는 파일만 처리
    if file_name.startswith("task-") and file_name.endswith(".png"):
        # "task-1680.jpg"에서 '1680' 부분만 추출
        file_id = file_name[len("task-"):9]  # "1680"

        # CSV에서 찾은 id_to_file_name과 매핑
        if file_id in id_to_file_name:
            new_name = id_to_file_name[file_id]  # "10.jpg"

            old_path = os.path.join(IMAGE_FOLDER, file_name)
            new_path = os.path.join(IMAGE_FOLDER, "label_"+ new_name)

            # 실제 파일 이름 변경
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_name}")
        else:
            print(f"ID {file_id} not found in CSV. Skipped: {file_name}")