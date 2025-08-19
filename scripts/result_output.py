import os
import json
import csv

def collect_results_to_csv(directory, output_csv_file):
    """
    遍历指定目录下的所有 "test_results" json 文件，并将结果汇总到 a CSV 文件中。

    参数:
        directory (str): 要搜索json文件的目录.
        output_csv_file (str): 输出的CSV文件名.
    """
    header_written = False
    all_data = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if "test_results" in filename and filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)

                # 第一次写入时，先写入CSV的表头
                if not header_written:
                    header = ["filename"] + list(data.keys())
                    all_data.append(header)
                    header_written = True

                # 提取数据并添加到列表中
                row = [filename] + list(data.values())
                all_data.append(row)

    # 将所有数据写入CSV文件
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_data)

    print(f"成功将数据写入到 {output_csv_file}")

# --- 使用方法 ---
# 1. 将下面的 'your_directory_path' 替换为你的实际路径
# 2. 运行这个脚本

directory_path = 'finetuned_checkpoints/crop_checkpoints/'
output_csv = 'test_results_summary.csv'

# 创建一个虚拟的目录和一些json文件来测试
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# 虚拟文件 1
data1 = {
    "test/loss": 1.0441, "test/Accuracy": 0.6361, "test/F1_Score": 0.5944, "test/mIoU": 0.4294
}
with open(os.path.join(directory_path, 'test_results_freeze_backbone.json'), 'w') as f:
    json.dump(data1, f)

# 虚拟文件 2
data2 = {
    "test/loss": 0.9876, "test/Accuracy": 0.7123, "test/F1_Score": 0.6876, "test/mIoU": 0.5123
}
with open(os.path.join(directory_path, 'test_results_full_finetune.json'), 'w') as f:
    json.dump(data2, f)

# 运行主函数
collect_results_to_csv(directory_path, output_csv)

if __name__ == "__main__":

    directory = "finetuned_checkpoints/crop_checkpoints"
    output_csv = "test_results_summary.csv"

    collect_results_to_csv(directory=directory, output_csv=output_csv)