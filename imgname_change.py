import os

IMAGE_FOLDER = '/home/user/dev/unet-landmark/input_raw'


for filename in os.listdir(IMAGE_FOLDER):
    # JPG 확장자를 가지는 파일만 처리 (대소문자 구분 없음)
    if filename.lower().endswith(".jpg"):
        # 기존 이름에서 확장자 분리
        base, ext = os.path.splitext(filename)  # 예: ("1", ".JPG")

        # 새 파일 이름 정의
        new_filename = f"input_{base}.jpg"  # "label_1.jpg"

        # 기존 경로와 새 경로를 구성
        old_path = os.path.join(IMAGE_FOLDER, filename)
        new_path = os.path.join(IMAGE_FOLDER, new_filename)

        # 실제 파일 이름 변경
        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_filename}")