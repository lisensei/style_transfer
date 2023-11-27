from torch.nn.modules.module import T
import torch
import torch.nn as nn
import torchvision.models as models
from pprint import pprint
from torchvision.models import VGG19_Weights
from copy import copy
import matplotlib.pyplot as plt
import torchvision.io as tio
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import os
torch.set_printoptions(precision=4)
weights=VGG19_Weights.DEFAULT
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transforms=T.Normalize(mean=mean,std=std)
model=models.vgg19(weights=weights)
for p in model.parameters():
  p.requires_grad=False


hard_sigmoid=nn.Hardsigmoid()
log_sigmoid=nn.LogSigmoid()
hard_tanh=nn.Hardtanh()
device="cuda" if torch.cuda.is_available() else "cpu"
model.cuda()
model.eval()

global sequential_outputs
sequential_outputs=[]

def change_pooling_layer(model):
  for i,f in enumerate(model.features):
    if isinstance(f,nn.MaxPool2d):
      model.features[i]=nn.AvgPool2d(f.kernel_size)

def compose(x,op="none"):
  if op=="tanh":
    return (torch.tanh(x)+1)/2
  elif op=="sigmoid":
    return torch.sigmoid(x)
  elif op=="hs":
    return hard_sigmoid(x)
  elif op=="ls":
    return log_sigmoid(x)
  elif op=="minmax":
    x=(x+torch.min(x))/torch.max(x)
    return x
  elif op=="relu":
    return torch.relu(x)
  elif op=="shrink":
    return torch.hardshrink(x)+0.5
  elif op=="ht":
    return hard_tanh(x)+1
  elif op=="sin":
     return torch.abs(torch.sin(x))
  else:
    return x
def forward_hook(self,input,output):
    sequential_outputs.append(output)


def clear_output_list(self,input):
  global sequential_outputs
  sequential_outputs.clear()

def compute_gram_matrix(channel_outputs):
  if channel_outputs.ndim!=4 and channel_outputs.ndim!=3:
    raise RuntimeError("inputs must have either 3 or 4 dimensions")
  if channel_outputs.ndim==4:
    channel_outputs=channel_outputs.squeeze(0).flatten(start_dim=1)
  gram=channel_outputs@(channel_outputs.permute(1,0))
  return gram,channel_outputs.numel()



model.features.register_forward_pre_hook(clear_output_list)
for n,m in model.features.named_modules():
  m.register_forward_hook(forward_hook)


content_image_path="/content/style_transfer_pics/photo.jpg"
style_image_path="/content/style_transfer_pics/style.jpg"
picname=content_image_path.split('/')[-1][:-4]
result_root=f"/content/drive/MyDrive/style_results/{picname}"
if not os.access(result_root,os.F_OK):
  os.makedirs(result_root)

content_image=Image.open(content_image_path)
content_image_size=[content_image.size[1],content_image.size[0]]
style_image=Image.open(style_image_path)
style_image_size=[style_image.size[1],style_image.size[0]]



content_resize_size=[300,300]
style_resize_size=[300,300]

content_image_tensor=TF.to_tensor(TF.resize(content_image,content_resize_size)).unsqueeze(0).to(device)
style_image_tensor=TF.to_tensor(TF.resize(style_image,style_resize_size)).unsqueeze(0).to(device)
#change_pooling_layer(model)
with torch.no_grad():
  model(compose(content_image_tensor))
  content_outputs=copy(sequential_outputs)
  model(compose(style_image_tensor))
  style_outputs=copy(sequential_outputs)

style_grams=[]
for out in style_outputs:
  style_gram,_=compute_gram_matrix(out)
  style_grams.append(style_gram)

#target_image=torch.randn((1,3,*content_resize_size),requires_grad=True,device=device)
target_image=content_image_tensor.clone()
target_image.requires_grad=True
epoch=4001

optimizer=torch.optim.LBFGS([target_image],lr=0.1)
style_loss_fn=nn.MSELoss()
content_loss_fn=nn.MSELoss()
all_relus=torch.tensor([1,3,6,8,11,13,15,17,20,22,24,26,27,29,31,33,35])
selected_style_layers=all_relus[[0,2,4,8,13]]

selected_content_layers=all_relus[[9]]
print(f"style activation layers: {selected_style_layers};content activation layers: {selected_content_layers}")

#weights=[0.001,1] worked
weights=[0.1,1]

for e in range(epoch):
  def closure():
    optimizer.zero_grad()
    with torch.no_grad():
      target_image.clamp_(0,1)
    model(compose(target_image))
    #compute style loss
    target_sequential_outputs=copy(sequential_outputs)
    style_loss=0
    for layer in selected_style_layers:
      target_gram,numel=compute_gram_matrix(target_sequential_outputs[layer])
      style_loss+=style_loss_fn(target_gram,style_grams[layer])/numel

    #compute content loss
    content_loss=0
    for l in selected_content_layers:
      content_loss+=content_loss_fn(target_sequential_outputs[l],content_outputs[l])
    loss=weights[0]*content_loss+weights[1]*style_loss/len(selected_style_layers)

    loss.backward(retain_graph=True)
    #print(f"epoch: {e};loss: {loss.detach().round(decimals=4)},unweighted content loss: {content_loss.detach().round(decimals=4)},unweighted style loss: {style_loss.detach().round(decimals=4)}")

    return loss
  optimizer.step(closure)
  if e%100==0:
    #print(f"epoch: {e};loss: {loss.detach().round(decimals=4)},unweighted content loss: {content_loss.detach().round(decimals=4)},unweighted style loss: {style_loss.detach().round(decimals=4)}")
    fig,axes=plt.subplots(1,3,constrained_layout=True)
    for a in axes:
      a.axis("off")
    generated_image=TF.to_pil_image(compose(target_image.detach()).squeeze(0).cpu())
    axes[0].imshow(content_image.resize(style_resize_size))
    axes[1].imshow(style_image.resize(style_resize_size))
    axes[2].imshow(generated_image)
    plt.show()
    generated_image.save(f"{result_root}/{picname}_{e}.jpg")
