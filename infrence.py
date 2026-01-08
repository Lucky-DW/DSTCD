import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import rasterio
from utils.metric import get_confusion_matrix, get_metric
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

y_transforms = transforms.Compose([
    transforms.ToTensor()
])

class SlideWindowDataset(Dataset):
    def __init__(self, img_a, img_b, window_size=256, stride=256):
        self.img_a = img_a
        self.img_b = img_b
        self.window_size = window_size
        self.stride = stride
        self.coords = []

        h, w = img_a.size[0], img_a.size[1]
        for i in range(0, w - window_size + 1, stride):
            for j in range(0, h - window_size + 1, stride):
                self.coords.append((i, j))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        box = (i, j, i + self.window_size, j + self.window_size)
        crop_a = x_transforms(self.img_a.crop(box))
        crop_b = x_transforms(self.img_b.crop(box))
        return crop_a, crop_b, (i, j)

def get_opt_sar_DCprediction_optimized(image_path1, image_path2, modelfile, img_size=256, batch_size=8):
    # 加载模型
    model = torch.load(modelfile, map_location=device)
    model.eval()
    with rasterio.open(image_path1) as ds:
        pre_img_A = ds.read([1, 2, 3])
        pre_img_A = np.array(pre_img_A)
        pre_img_A = np.uint8(pre_img_A)
        pre_img_A = Image.fromarray(np.transpose(pre_img_A, (1, 2, 0)))

    with rasterio.open(image_path2) as ds:
        pre_img_B = ds.read([1])
        pre_img_B = np.array(pre_img_B)
        pre_img_B = np.uint8(pre_img_B)
        pre_img_B = Image.fromarray(np.squeeze(pre_img_B)).convert('RGB')

    w = pre_img_A.size[0]  # 待预测影像宽
    h = pre_img_A.size[1]  # 待预测影像高

    stride = img_size
    padding_h = int(h / img_size) * img_size + img_size + stride
    padding_w = int(w / img_size) * img_size + img_size + stride
    padding_result = np.zeros((padding_h, padding_w), dtype=np.uint8)

    coords = []
    for i in range(0, w, stride):
        for j in range(0, h, stride):
            coords.append((i, j))

    with torch.no_grad():
        for start_idx in range(0, len(coords), batch_size):
            end_idx = min(start_idx + batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]

            batch_a = []
            batch_b = []

            for i, j in batch_coords:
                box = (i, j, i + img_size, j + img_size)
                crop_a = x_transforms(pre_img_A.crop(box))
                crop_b = x_transforms(pre_img_B.crop(box))
                batch_a.append(crop_a)
                batch_b.append(crop_b)
            batch_a = torch.stack(batch_a).to(device)
            batch_b = torch.stack(batch_b).to(device)

            # 批量推理
            _, pre_data = model(batch_a, batch_b)
            mask_pre = (pre_data > 0.5).float()

            mask_pre_np = mask_pre.cpu().numpy()

            for idx, (i, j) in enumerate(batch_coords):
                img_pre = np.squeeze(mask_pre_np[idx])
                padding_result[j:j+img_size, i:i+img_size] += (img_pre * 255).astype(np.uint8)

    result = padding_result[0:h, 0:w]
    result = result

    return result


if __name__ == "__main__":

    dataset_list = ['xinxiang', 'noto']
    
    for pth_name in dataset_list:
        print(f"Processing {pth_name}")
        dataset_name = pth_name
        pth_path = r'result/' + pth_name
        path = './data/' + dataset_name
        image_path1 = path + r'/t1/' + dataset_name + '.tif'
        image_path2 = path + r'/t2/' + dataset_name + '.tif'
        true_path = path + r'/gt/' + dataset_name + '.tif'
        modelfile = pth_path + r'/best_model.pth'
        save_path = pth_path + r'/' + dataset_name + '.tif'
        true_img = np.array(Image.open(true_path))
        out = get_opt_sar_DCprediction_optimized(
            image_path1, image_path2, modelfile, 
            img_size=256, batch_size=8
        )
        Image.fromarray(out).save(save_path)
        confusion_matrix = get_confusion_matrix(out, true_img)
        accuracy, f1_score, iou, precision, recall = get_metric(confusion_matrix)
        
        print(f"Metrics: Accuracy={accuracy:.4f}, F1={f1_score:.4f}, "
                f"IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")