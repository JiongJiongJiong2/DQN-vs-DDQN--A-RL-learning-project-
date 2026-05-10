"""
将 results 目录下带 -test 标签的文件夹中的 PDF 文件转换为同名的 JPG 文件。
依赖: pdf2image, Pillow, poppler
"""
from pdf2image import convert_from_path
from pathlib import Path

# 需要处理的 test 文件夹
test_folders = [
    "results/260510-2-A-DDQN-test",
    "results/260510-2-A-DQN-test",
    "results/260510-2-C-DDQN-test",
    "results/260510-2-C-DQN-test",
]

converted = 0
skipped = 0

for folder in test_folders:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"跳过: {folder} (文件夹不存在)")
        continue

    for pdf_file in sorted(folder_path.glob("*.pdf")):
        jpg_path = pdf_file.with_suffix('.jpg')

        # 如果 JPG 已存在且比 PDF 新，跳过
        if jpg_path.exists() and jpg_path.stat().st_mtime > pdf_file.stat().st_mtime:
            print(f"跳过 (JPG已存在): {jpg_path}")
            skipped += 1
            continue

        try:
            # 将 PDF 第一页转换为 JPG，DPI=150 保证清晰度
            images = convert_from_path(str(pdf_file), dpi=150)
            images[0].save(str(jpg_path), 'JPEG', quality=95)
            print(f"转换成功: {pdf_file} -> {jpg_path}")
            converted += 1
        except Exception as e:
            print(f"转换失败: {pdf_file} | 错误: {e}")

print(f"\n完成! 转换 {converted} 个文件, 跳过 {skipped} 个文件。")