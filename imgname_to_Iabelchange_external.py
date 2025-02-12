import os

IMAGE_FOLDER = '/Users/neverthe1ess/PycharmProjects/unet-landmark/external_label_raw'

id = 561

for filename in os.listdir(IMAGE_FOLDER):
    # JPG 확장자를 가지는 파일만 처리 (대소문자 구분 없음)
    if filename.lower().endswith(".jpg"):
        # 새 파일 이름 정의
        new_filename = f"label_{id}.jpg"  # "label_1.jpg"

        # 기존 경로와 새 경로를 구성
        old_path = os.path.join(IMAGE_FOLDER, filename)
        new_path = os.path.join(IMAGE_FOLDER, new_filename)

        # 실제 파일 이름 변경
        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_filename}")

    id = id + 1