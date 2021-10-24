import os
import cv2
import torch
from torchvision import transforms
from model import CNNLSTM
from PIL import Image
from torch.autograd import Variable
from collections import deque
import warnings

def getSizeVideo(file_path):
    vid = cv2.VideoCapture(file_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return width, height
warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

video_path = r"video\V_106.mp4"
width, height = getSizeVideo(video_path)
print(width, height)

cap = cv2.VideoCapture(video_path)

seq_length = 15

frames = deque(maxlen=seq_length)
count = 0

model = CNNLSTM().to(device)
model.load_state_dict(torch.load("save/model_best25000.pth", map_location=torch.device('cpu')))
model.eval()


out = cv2.VideoWriter('result/25000fightV106.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,360))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if len(frames) < seq_length:
        if count % 5 == 0:
            frames.append(frame)
    else:
        tensor_frames = torch.tensor(())

        for j in frames:
            img = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            im_pil = im_pil.resize((224, 224))
            im_pil = transform(im_pil)
            im_pil = Variable(torch.unsqueeze(im_pil, 0))

            tensor_frames = torch.cat((tensor_frames, im_pil), 0)

        tensor_frames = tensor_frames.to(device)
        tensor_frames = torch.unsqueeze(tensor_frames, 0)

        with torch.no_grad():
            pred = model(tensor_frames)
        pred_idx = torch.argmax(pred, dim=1)[0]

        if pred_idx:
            cv2.putText(frame, "noFight", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),
                        lineType=cv2.LINE_AA, thickness=2)
        else:
            cv2.putText(frame, "Fight", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),
                        lineType=cv2.LINE_AA, thickness=2)

        frames.popleft()

    count += 1

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
out.release()
cv2.destroyAllWindows()
