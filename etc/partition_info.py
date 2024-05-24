import torch

# label.pt 파일 로드
file_path = './data/partitions/ogbn-products/2-parts/ogbn-products-partitions/part_0/edge_feats.pt'
data = torch.load(file_path)

# 데이터 출력
print("Data:", data)

tensor_size = data.size()
print("Tensor size:", tensor_size)

# 텍스트 파일로 변환
txt_file_path = 'val_partition1.txt'
with open(txt_file_path, 'w') as f:
    f.write(f"Tensor size: {tensor_size}\n")
    f.write("Tensor data:\n")
    for value in data:
        f.write(f"{value.item()}\n")

print(f"Data has been saved to {txt_file_path}")
